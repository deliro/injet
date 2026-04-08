use ed25519_dalek::{Signature, Signer as _, SigningKey, VerifyingKey};

use crate::auth::error::KeyError;

/// Sign `auth_input` with an Ed25519 signing key.
#[must_use]
pub fn sign_ed25519(key: &SigningKey, auth_input: &[u8]) -> [u8; 64] {
    key.sign(auth_input).to_bytes()
}

/// Verify an Ed25519 signature using `verify_strict` to reject malleable encodings.
#[must_use]
pub fn verify_ed25519(pubkey: &VerifyingKey, auth_input: &[u8], sig: &[u8; 64]) -> bool {
    let signature = Signature::from_bytes(sig);
    pubkey.verify_strict(auth_input, &signature).is_ok()
}

/// Build a `SigningKey` from a raw 32-byte seed.
#[must_use]
pub fn signing_key_from_seed(seed: &[u8; 32]) -> SigningKey {
    SigningKey::from_bytes(seed)
}

/// Build a `VerifyingKey` from raw 32 bytes.
///
/// # Errors
///
/// Returns [`KeyError::UnsupportedAlgorithm`] if the bytes do not decode to a
/// valid Ed25519 public key.
pub fn verifying_key_from_bytes(bytes: &[u8; 32]) -> Result<VerifyingKey, KeyError> {
    VerifyingKey::from_bytes(bytes).map_err(|_| KeyError::UnsupportedAlgorithm)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{sign_ed25519, signing_key_from_seed, verify_ed25519, verifying_key_from_bytes};

    // RFC 8032 test vector 1: empty message.
    const RFC8032_SEED: [u8; 32] = [
        0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c,
        0xc4, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae,
        0x7f, 0x60,
    ];
    const RFC8032_PUB: [u8; 32] = [
        0xd7, 0x5a, 0x98, 0x01, 0x82, 0xb1, 0x0a, 0xb7, 0xd5, 0x4b, 0xfe, 0xd3, 0xc9, 0x64, 0x07,
        0x3a, 0x0e, 0xe1, 0x72, 0xf3, 0xda, 0xa6, 0x23, 0x25, 0xaf, 0x02, 0x1a, 0x68, 0xf7, 0x07,
        0x51, 0x1a,
    ];

    #[test]
    fn rfc8032_vector_1() {
        let signing = signing_key_from_seed(&RFC8032_SEED);
        let verifying = signing.verifying_key();
        assert_eq!(verifying.to_bytes(), RFC8032_PUB);
        let sig = sign_ed25519(&signing, b"");
        assert!(verify_ed25519(&verifying, b"", &sig));
    }

    #[test]
    fn verify_rejects_bit_flipped_signature() {
        let signing = signing_key_from_seed(&[0x42_u8; 32]);
        let verifying = signing.verifying_key();
        let mut sig = sign_ed25519(&signing, b"hello");
        sig[0] ^= 0x01;
        assert!(!verify_ed25519(&verifying, b"hello", &sig));
    }

    #[test]
    fn verify_rejects_message_tamper() {
        let signing = signing_key_from_seed(&[0x42_u8; 32]);
        let verifying = signing.verifying_key();
        let sig = sign_ed25519(&signing, b"hello");
        assert!(!verify_ed25519(&verifying, b"hello!", &sig));
    }

    #[test]
    fn verify_rejects_wrong_key() {
        let signing_a = signing_key_from_seed(&[0xAA_u8; 32]);
        let signing_b = signing_key_from_seed(&[0xBB_u8; 32]);
        let sig = sign_ed25519(&signing_a, b"data");
        assert!(!verify_ed25519(&signing_b.verifying_key(), b"data", &sig));
    }

    #[test]
    fn verifying_key_from_bytes_rejects_garbage() {
        // y = 0x_AB00_01 (little-endian) is not on the Ed25519 curve.
        let mut bad = [0x00_u8; 32];
        bad[0] = 0x01;
        bad[2] = 0xAB;
        assert!(verifying_key_from_bytes(&bad).is_err());
    }
}
