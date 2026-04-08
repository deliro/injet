use crate::auth::error::KeyError;

const ARGON2_M_KIB: u32 = 64 * 1024;
const ARGON2_T: u32 = 3;
const ARGON2_P: u32 = 4;

/// Derive a 32-byte BLAKE3 keyed-hash key from a passphrase using Argon2id.
///
/// Parameters are the OWASP "interactive" baseline as of 2026
/// (m=64 MiB, t=3, p=4) and are intentionally hard-coded — changing them is
/// wire-incompatible with previously written containers.
///
/// # Errors
///
/// Returns [`KeyError::Argon2Params`] or [`KeyError::Argon2Hash`] if the
/// underlying argon2 crate rejects the parameters or fails to compute the hash.
pub fn derive_psk(passphrase: &[u8], salt: &[u8; 16]) -> Result<[u8; 32], KeyError> {
    let params = argon2::Params::new(ARGON2_M_KIB, ARGON2_T, ARGON2_P, Some(32))
        .map_err(|_| KeyError::Argon2Params)?;
    let argon2 = argon2::Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);
    let mut out = [0_u8; 32];
    argon2
        .hash_password_into(passphrase, salt, &mut out)
        .map_err(|_| KeyError::Argon2Hash)?;
    Ok(out)
}

/// Compute a BLAKE3 keyed-hash MAC over `auth_input`.
#[must_use]
pub fn psk_mac(key: &[u8; 32], auth_input: &[u8]) -> [u8; 32] {
    *blake3::keyed_hash(key, auth_input).as_bytes()
}

/// Generate a fresh 16-byte salt from the OS RNG.
///
/// # Errors
///
/// Returns [`KeyError::Rng`] if `getrandom` fails.
pub fn fresh_salt() -> Result<[u8; 16], KeyError> {
    let mut salt = [0_u8; 16];
    getrandom::getrandom(&mut salt).map_err(|_| KeyError::Rng)?;
    Ok(salt)
}

/// Verify a PSK MAC in constant time.
#[must_use]
pub fn verify_psk(key: &[u8; 32], auth_input: &[u8], expected_mac: &[u8; 32]) -> bool {
    let computed = psk_mac(key, auth_input);
    constant_time_eq(&computed, expected_mac)
}

fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

use crate::meta::{AuthField, Signer};

pub struct PskSigner {
    key: [u8; 32],
    salt: [u8; 16],
}

impl PskSigner {
    /// Build a signer from a passphrase. Generates a fresh salt and derives
    /// the Argon2id key.
    ///
    /// # Errors
    ///
    /// Returns [`KeyError::Rng`] if the OS RNG fails, or
    /// [`KeyError::Argon2Params`] / [`KeyError::Argon2Hash`] from the KDF.
    pub fn new(passphrase: &[u8]) -> Result<Self, KeyError> {
        let salt = fresh_salt()?;
        let key = derive_psk(passphrase, &salt)?;
        Ok(Self { key, salt })
    }
}

impl Signer for PskSigner {
    fn sign(&self, auth_input: &[u8]) -> AuthField {
        let mac = psk_mac(&self.key, auth_input);
        AuthField::Mac {
            salt: self.salt,
            mac,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{derive_psk, fresh_salt, psk_mac, verify_psk};
    use rstest::rstest;
    use std::collections::HashSet;

    #[test]
    fn derive_psk_is_deterministic() {
        let salt = [0x42_u8; 16];
        let k1 = derive_psk(b"correct horse battery staple", &salt).unwrap();
        let k2 = derive_psk(b"correct horse battery staple", &salt).unwrap();
        assert_eq!(k1, k2);
        insta::assert_debug_snapshot!("derive_psk_known", k1);
    }

    #[test]
    fn derive_psk_different_salt_different_key() {
        let k1 = derive_psk(b"hunter2", &[1_u8; 16]).unwrap();
        let k2 = derive_psk(b"hunter2", &[2_u8; 16]).unwrap();
        assert_ne!(k1, k2);
    }

    #[rstest]
    #[case::empty(b"")]
    #[case::short(b"hello")]
    #[case::longer(&[0xAB_u8; 1024])]
    fn psk_mac_snapshots(#[case] input: &[u8]) {
        let key = [0x11_u8; 32];
        let mac = psk_mac(&key, input);
        let name = format!("psk_mac_len_{}", input.len());
        insta::assert_debug_snapshot!(name, mac);
    }

    #[test]
    fn verify_psk_accepts_correct_mac() {
        let key = [0x77_u8; 32];
        let mac = psk_mac(&key, b"hello world");
        assert!(verify_psk(&key, b"hello world", &mac));
    }

    #[test]
    fn verify_psk_rejects_tampered_mac() {
        let key = [0x77_u8; 32];
        let mut mac = psk_mac(&key, b"hello world");
        mac[0] ^= 0x01;
        assert!(!verify_psk(&key, b"hello world", &mac));
    }

    #[test]
    fn fresh_salt_returns_distinct_values() {
        let mut seen: HashSet<[u8; 16]> = HashSet::new();
        for _ in 0_i32..100_i32 {
            let s = fresh_salt().unwrap();
            assert!(seen.insert(s), "duplicate salt from RNG");
        }
    }
}
