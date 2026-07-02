//! Global model state held by the syncer: fragment layout, parameters Θ,
//! and outer-optimizer momentum, all in f32.

use anyhow::{bail, Result};

use crate::merge;
use crate::protocol::Reader;

pub const MERGE_AVG: u8 = 0;
pub const MERGE_RDA: u8 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct FragmentInfo {
    pub merge_mode: u8,
    pub tensor_numels: Vec<u64>,
}

impl FragmentInfo {
    pub fn numel(&self) -> usize {
        self.tensor_numels.iter().sum::<u64>() as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub fragments: Vec<FragmentInfo>,
}

impl Layout {
    /// Decode the layout section of a HELLO payload (cursor positioned after
    /// learner_id/dtype/num_fragments have been consumed except the count).
    pub fn decode(r: &mut Reader<'_>, num_fragments: u32) -> Result<Self> {
        let mut fragments = Vec::with_capacity(num_fragments as usize);
        for _ in 0..num_fragments {
            let merge_mode = r.u8()?;
            if merge_mode > MERGE_RDA {
                bail!("bad merge mode {merge_mode}");
            }
            let num_tensors = r.u32()?;
            let mut tensor_numels = Vec::with_capacity(num_tensors as usize);
            for _ in 0..num_tensors {
                tensor_numels.push(r.u64()?);
            }
            fragments.push(FragmentInfo { merge_mode, tensor_numels });
        }
        Ok(Layout { fragments })
    }
}

pub struct GlobalState {
    pub layout: Layout,
    /// Θ_p, flat f32 per fragment (concatenated tensors in layout order).
    pub params: Vec<Vec<f32>>,
    /// Nesterov momentum buffers, same shape as params.
    momentum: Vec<Vec<f32>>,
    pub initialized: Vec<bool>,
    pub outer_lr: f32,
    pub outer_momentum: f32,
    /// Dtype used on the wire (from HELLO); merge math stays f32.
    pub wire_dtype: u8,
}

impl GlobalState {
    pub fn new(layout: Layout, outer_lr: f32, outer_momentum: f32, wire_dtype: u8) -> Self {
        let params: Vec<Vec<f32>> = layout.fragments.iter().map(|f| vec![0.0; f.numel()]).collect();
        let momentum = params.clone();
        let initialized = vec![false; layout.fragments.len()];
        Self { layout, params, momentum, initialized, outer_lr, outer_momentum, wire_dtype }
    }

    pub fn all_initialized(&self) -> bool {
        self.initialized.iter().all(|&b| b)
    }

    pub fn init_fragment(&mut self, fid: usize, values: Vec<f32>) -> Result<()> {
        if values.len() != self.params[fid].len() {
            bail!("init fragment {fid}: got {} values, expected {}", values.len(), self.params[fid].len());
        }
        if !self.initialized[fid] {
            self.params[fid] = values;
            self.initialized[fid] = true;
        }
        Ok(())
    }

    /// Merge learner copies of fragment `fid` and apply the outer step.
    /// Returns the l2 norm of the merged outer gradient (for logging).
    pub fn merge_and_step(&mut self, fid: usize, learners: &[&[f32]], weights: &[f64]) -> Result<f64> {
        let frag = &self.layout.fragments[fid];
        let numel = frag.numel();
        for (i, l) in learners.iter().enumerate() {
            if l.len() != numel {
                bail!("push for fragment {fid} from entry {i} has {} values, expected {numel}", l.len());
            }
        }
        let anchor = &self.params[fid];
        let mut delta = vec![0.0f32; numel];
        // Merge per tensor slice within the fragment.
        let mut off = 0usize;
        for &tn in &frag.tensor_numels {
            let tn = tn as usize;
            let slice_learners: Vec<&[f32]> = learners.iter().map(|l| &l[off..off + tn]).collect();
            let out = &mut delta[off..off + tn];
            match frag.merge_mode {
                MERGE_AVG => merge::merge_avg(&anchor[off..off + tn], &slice_learners, weights, out),
                _ => merge::merge_rda(&anchor[off..off + tn], &slice_learners, weights, out),
            }
            off += tn;
        }
        let gnorm = delta.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        merge::nesterov_step(
            &mut self.params[fid],
            &mut self.momentum[fid],
            &delta,
            self.outer_lr,
            self.outer_momentum,
        );
        Ok(gnorm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout2() -> Layout {
        Layout {
            fragments: vec![
                FragmentInfo { merge_mode: MERGE_AVG, tensor_numels: vec![4] },
                FragmentInfo { merge_mode: MERGE_RDA, tensor_numels: vec![2, 2] },
            ],
        }
    }

    #[test]
    fn init_once() {
        let mut st = GlobalState::new(layout2(), 0.7, 0.9, crate::protocol::DTYPE_F32);
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(0, vec![2.0; 4]).unwrap(); // ignored
        assert_eq!(st.params[0], vec![1.0; 4]);
        assert!(!st.all_initialized());
        st.init_fragment(1, vec![0.0; 4]).unwrap();
        assert!(st.all_initialized());
    }

    #[test]
    fn merge_moves_toward_learners() {
        let mut st = GlobalState::new(layout2(), 1.0, 0.0, crate::protocol::DTYPE_F32); // plain SGD lr=1 = weight averaging
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![1.0; 4]).unwrap();
        let learner = vec![0.0f32; 4];
        let g = st.merge_and_step(0, &[&learner], &[1.0]).unwrap();
        assert!(g > 0.0);
        // Θ − 1.0·(Θ − θ) = θ
        assert_eq!(st.params[0], vec![0.0; 4]);
    }

    #[test]
    fn size_mismatch_rejected() {
        let mut st = GlobalState::new(layout2(), 0.7, 0.9, crate::protocol::DTYPE_F32);
        assert!(st.init_fragment(0, vec![1.0; 3]).is_err());
        let learner = vec![0.0f32; 3];
        assert!(st.merge_and_step(0, &[&learner], &[1.0]).is_err());
    }
}
