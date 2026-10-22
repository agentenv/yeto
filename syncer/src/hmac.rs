//! HMAC-SHA256 (RFC 2104) for the island HELLO MAC.

use sha2::{Digest, Sha256};

pub const HMAC_LEN: usize = 32;

pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    const B: usize = 64;
    let mut k = [0u8; B];
    if key.len() > B {
        k[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(k.map(|b| b ^ 0x36));
    inner.update(msg);
    let ih = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(k.map(|b| b ^ 0x5c));
    outer.update(ih);
    outer.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_rfc4231_case_2() {
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        let hex: String = mac.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    }
}
