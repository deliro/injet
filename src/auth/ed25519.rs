use std::path::{Path, PathBuf};

use ed25519_dalek::{Signature, Signer as _, SigningKey, VerifyingKey};
use ssh_key::{Algorithm, PrivateKey, PublicKey};
use zeroize::Zeroizing;

use crate::auth::error::KeyError;
use crate::auth::passphrase::PassphraseSource;

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

/// Load an OpenSSH Ed25519 private key from disk.
///
/// If the key is encrypted, decrypts it using the supplied passphrase
/// source. If `encrypted_key_passphrase` is `None` and the key is
/// encrypted, falls back to a TTY prompt; if no TTY is available the
/// returned error is [`KeyError::EncryptedKeyNoPassphrase`].
///
/// # Errors
///
/// Returns [`KeyError`] variants for I/O failures, parse failures,
/// non-Ed25519 algorithms, and decryption failures.
pub fn load_signing_key(
    path: &Path,
    encrypted_key_passphrase: Option<&PassphraseSource>,
) -> Result<SigningKey, KeyError> {
    let pem = std::fs::read(path).map_err(|source| KeyError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut private = PrivateKey::from_openssh(&pem).map_err(|source| KeyError::OpenSshParse {
        path: path.to_path_buf(),
        source,
    })?;
    if private.is_encrypted() {
        let pass = if let Some(src) = encrypted_key_passphrase {
            crate::auth::passphrase::load(src)?
        } else {
            if !is_tty() {
                return Err(KeyError::EncryptedKeyNoPassphrase);
            }
            crate::auth::passphrase::load(&PassphraseSource::Prompt)?
        };
        private = private.decrypt(&pass).map_err(|_| KeyError::DecryptFailed)?;
    }
    if private.algorithm() != Algorithm::Ed25519 {
        return Err(KeyError::UnsupportedAlgorithm);
    }
    let keypair = private
        .key_data()
        .ed25519()
        .ok_or(KeyError::UnsupportedAlgorithm)?;
    let seed_bytes: [u8; 32] = keypair.private.to_bytes();
    let seed = Zeroizing::new(seed_bytes);
    Ok(signing_key_from_seed(&seed))
}

/// Load an OpenSSH Ed25519 public key from a file.
///
/// # Errors
///
/// Returns [`KeyError`] for I/O, parse, or algorithm errors.
pub fn load_verifying_key_from_file(path: &Path) -> Result<VerifyingKey, KeyError> {
    let public = PublicKey::read_openssh_file(path).map_err(|source| KeyError::OpenSshParse {
        path: path.to_path_buf(),
        source,
    })?;
    public_key_to_verifying(&public, path)
}

/// Parse an `ssh-ed25519 AAAA... [comment]` line into a `VerifyingKey`.
///
/// # Errors
///
/// Returns parse / algorithm errors as [`KeyError`].
pub fn parse_verifying_key(line: &str) -> Result<VerifyingKey, KeyError> {
    let public = PublicKey::from_openssh(line.trim()).map_err(|source| KeyError::OpenSshParse {
        path: PathBuf::new(),
        source,
    })?;
    public_key_to_verifying(&public, Path::new(""))
}

fn public_key_to_verifying(public: &PublicKey, _origin: &Path) -> Result<VerifyingKey, KeyError> {
    if public.algorithm() != Algorithm::Ed25519 {
        return Err(KeyError::UnsupportedAlgorithm);
    }
    let ed = public
        .key_data()
        .ed25519()
        .ok_or(KeyError::UnsupportedAlgorithm)?;
    let bytes: [u8; 32] = ed.0;
    verifying_key_from_bytes(&bytes)
}

fn is_tty() -> bool {
    use std::io::IsTerminal as _;
    std::io::stdin().is_terminal()
}

use crate::meta::{AuthField, Signer};

pub struct Ed25519Signer {
    inner: SigningKey,
}

impl Ed25519Signer {
    #[must_use]
    pub fn new(inner: SigningKey) -> Self {
        Self { inner }
    }
}

impl Signer for Ed25519Signer {
    fn sign(&self, auth_input: &[u8]) -> AuthField {
        AuthField::Signature(sign_ed25519(&self.inner, auth_input))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::path::PathBuf;

    use super::{
        load_signing_key, load_verifying_key_from_file, parse_verifying_key, sign_ed25519,
        signing_key_from_seed, verify_ed25519, verifying_key_from_bytes,
    };
    use crate::auth::error::KeyError;

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

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/keys")
            .join(name)
    }

    #[test]
    fn load_unencrypted_private_key() {
        let key = load_signing_key(&fixture("signer_ed25519"), None).unwrap();
        let sig = sign_ed25519(&key, b"data");
        assert!(verify_ed25519(&key.verifying_key(), b"data", &sig));
    }

    #[test]
    fn load_public_key_file() {
        let pubkey = load_verifying_key_from_file(&fixture("signer_ed25519.pub")).unwrap();
        let signing = load_signing_key(&fixture("signer_ed25519"), None).unwrap();
        assert_eq!(pubkey.to_bytes(), signing.verifying_key().to_bytes());
    }

    #[test]
    fn parse_public_key_line() {
        let line = std::fs::read_to_string(fixture("signer_ed25519.pub")).unwrap();
        let pubkey = parse_verifying_key(&line).unwrap();
        assert_eq!(pubkey.to_bytes().len(), 32);
    }

    #[test]
    fn load_encrypted_private_key_with_passphrase_file() {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"test-passphrase\n").unwrap();
        let src = crate::auth::passphrase::PassphraseSource::File(f.path().to_path_buf());
        let key = load_signing_key(&fixture("signer_ed25519_encrypted"), Some(&src)).unwrap();
        let sig = sign_ed25519(&key, b"data");
        assert!(verify_ed25519(&key.verifying_key(), b"data", &sig));
    }

    #[test]
    fn load_encrypted_private_key_wrong_passphrase() {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"wrong\n").unwrap();
        let src = crate::auth::passphrase::PassphraseSource::File(f.path().to_path_buf());
        let err =
            load_signing_key(&fixture("signer_ed25519_encrypted"), Some(&src)).unwrap_err();
        assert!(matches!(err, KeyError::DecryptFailed));
    }

    #[test]
    fn unsupported_algorithm_rejected() {
        let rsa_pub = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQ== test";
        let res = parse_verifying_key(rsa_pub);
        assert!(matches!(
            res,
            Err(KeyError::OpenSshParse { .. } | KeyError::UnsupportedAlgorithm)
        ));
    }
}
