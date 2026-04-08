pub mod classify;
pub mod ed25519;
pub mod error;
pub mod passphrase;
pub mod psk;

pub use classify::{classify, AuthAlgorithm, ExtractDecision};
pub use ed25519::Ed25519Signer;
pub use error::{AuthError, KeyError};
pub use psk::PskSigner;

use std::path::PathBuf;

use ed25519_dalek::VerifyingKey;
use zeroize::Zeroizing;

use crate::auth::passphrase::PassphraseSource;
use crate::meta::MetaField;

#[derive(Debug, Clone)]
pub enum AuthSpec {
    Psk(PassphraseSource),
    Ed25519 {
        key_path: PathBuf,
        key_passphrase: Option<PassphraseSource>,
    },
}

#[derive(Debug, Clone)]
pub enum VerifySpec {
    Psk(PassphraseSource),
    Ed25519Path(PathBuf),
    Ed25519Inline(String),
}

impl AuthSpec {
    #[must_use]
    pub fn algorithm(&self) -> AuthAlgorithm {
        match self {
            AuthSpec::Psk(_) => AuthAlgorithm::Psk,
            AuthSpec::Ed25519 { .. } => AuthAlgorithm::Ed25519,
        }
    }
}

impl VerifySpec {
    #[must_use]
    pub fn algorithm(&self) -> AuthAlgorithm {
        match self {
            VerifySpec::Psk(_) => AuthAlgorithm::Psk,
            VerifySpec::Ed25519Path(_) | VerifySpec::Ed25519Inline(_) => AuthAlgorithm::Ed25519,
        }
    }
}

pub enum VerifyJob {
    Psk {
        passphrase: Zeroizing<Vec<u8>>,
        salt: [u8; 16],
        mac: [u8; 32],
    },
    Ed25519 {
        pubkey: VerifyingKey,
        sig: [u8; 64],
    },
}

#[must_use]
pub fn detect_auth(fields: &[MetaField]) -> Option<AuthAlgorithm> {
    fields.iter().find_map(|f| match f {
        MetaField::Mac { .. } => Some(AuthAlgorithm::Psk),
        MetaField::Signature(_) => Some(AuthAlgorithm::Ed25519),
        _ => None,
    })
}

/// Pair the user-supplied verification key source with the auth field
/// found in the container, returning a typed verification job.
///
/// # Errors
///
/// Returns [`AuthError::MissingAuthField`] if no matching auth TLV is in
/// `fields`, or [`AuthError::VerificationFailed`] if key loading fails.
pub fn pair_up(spec: &VerifySpec, fields: &[MetaField]) -> Result<VerifyJob, AuthError> {
    match spec {
        VerifySpec::Psk(src) => {
            let (salt, mac) = fields
                .iter()
                .find_map(|f| match f {
                    MetaField::Mac { salt, mac } => Some((*salt, *mac)),
                    _ => None,
                })
                .ok_or(AuthError::MissingAuthField)?;
            let passphrase = passphrase::load(src).map_err(|_| AuthError::VerificationFailed)?;
            Ok(VerifyJob::Psk {
                passphrase,
                salt,
                mac,
            })
        }
        VerifySpec::Ed25519Path(path) => {
            let sig = signature_from_fields(fields)?;
            let pubkey = ed25519::load_verifying_key_from_file(path)
                .map_err(|_| AuthError::VerificationFailed)?;
            Ok(VerifyJob::Ed25519 { pubkey, sig })
        }
        VerifySpec::Ed25519Inline(line) => {
            let sig = signature_from_fields(fields)?;
            let pubkey =
                ed25519::parse_verifying_key(line).map_err(|_| AuthError::VerificationFailed)?;
            Ok(VerifyJob::Ed25519 { pubkey, sig })
        }
    }
}

fn signature_from_fields(fields: &[MetaField]) -> Result<[u8; 64], AuthError> {
    fields
        .iter()
        .find_map(|f| match f {
            MetaField::Signature(sig) => Some(*sig),
            _ => None,
        })
        .ok_or(AuthError::MissingAuthField)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{detect_auth, AuthAlgorithm};
    use crate::meta::MetaField;

    #[test]
    fn detect_auth_finds_mac() {
        let fields = vec![MetaField::Mac {
            salt: [0_u8; 16],
            mac: [0_u8; 32],
        }];
        assert_eq!(detect_auth(&fields), Some(AuthAlgorithm::Psk));
    }

    #[test]
    fn detect_auth_finds_signature() {
        let fields = vec![MetaField::Signature([0_u8; 64])];
        assert_eq!(detect_auth(&fields), Some(AuthAlgorithm::Ed25519));
    }

    #[test]
    fn detect_auth_returns_none_for_unsigned() {
        let fields = vec![MetaField::Size(7)];
        assert_eq!(detect_auth(&fields), None);
    }
}
