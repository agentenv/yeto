import json
from pathlib import Path
import tempfile
import unittest

from . import checkpoint


class SnapshotTests(unittest.TestCase):
    def test_exact_update_budget_requires_complete_history(self):
        rows=[json.dumps({'event':'verified_optimizer_update','step':i,
            'measurements':{'num_label_tokens':float(i+1)}}) for i in range(25)]
        self.assertEqual(checkpoint.budget_from_events('\n'.join(rows),25),
            {'cumulative_supervised_tokens':325,'cumulative_input_tokens':None})
        with self.assertRaisesRegex(ValueError,'incomplete'):
            checkpoint.budget_from_events('\n'.join(rows[1:]),25)
        with self.assertRaisesRegex(ValueError,'changed'):
            checkpoint.budget_from_events('\n'.join(rows+[rows[0].replace('1.0','2.0')]),25)

    def fixture(self,root):
        source=root/'checkpoints/epoch_0_step_24';(source/'model').mkdir(parents=True)
        for name in ['.metadata',*[f'__{i}_0.distcp' for i in range(8)]]:
            (source/'model'/name).write_text(name)
        for name in ['step_scheduler.pt','resume_guard.pt']:(source/name).write_text(name)
        config=root/'train.json';config.write_text(json.dumps({'dataset':{'_target_':
            'training.qwen38_turn_boundary_v2.data.TurnTokenDataset'},
            'checkpoint':{'checkpoint_dir':str(source.parent)}}))
        return source,config

    def test_snapshot_survives_source_retention_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();source,config=self.fixture(root);dest=root/'quality/source'
            identity=checkpoint.snapshot(source,dest,arm='no-cot-turn-v2',completed_updates=25,
                budget={'cumulative_supervised_tokens':100},config=config)
            self.assertEqual(checkpoint.verify_snapshot(dest),identity)
            self.assertFalse((dest/'optim').exists())
            for p in source.rglob('*'):
                if p.is_file():p.unlink()
            self.assertEqual(checkpoint.verify_snapshot(dest),identity)
            (dest/'model/__0_0.distcp').write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):checkpoint.verify_snapshot(dest)

    def test_wrong_arm_and_incomplete_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();source,config=self.fixture(root)
            args=dict(completed_updates=25,budget={'cumulative_supervised_tokens':100},config=config)
            with self.assertRaisesRegex(ValueError,'requested corrected arm'):
                checkpoint.snapshot(source,root/'quality/a',arm='masked-cot-native-gap-v3',**args)
            (source/'.incomplete').touch()
            with self.assertRaisesRegex(ValueError,'complete'):
                checkpoint.snapshot(source,root/'quality/b',arm='no-cot-turn-v2',**args)


if __name__=='__main__':unittest.main()
