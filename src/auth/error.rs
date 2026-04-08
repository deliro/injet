use std::path::PathBuf;

use thiserror::Error;

use crate::auth::classify::AuthAlgorithm;

#[derive(Debug, Error)]
pub enum KeyError {
    #[error("failed to read key file {path}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("environment variable {0} is not set")]
    EnvVarMissing(String),
    #[error("passphrase is empty")]
    EmptyPassphrase,
    #[error("passphrase exceeds 4096 bytes")]
    PassphraseTooLong,
    #[error("failed to parse OpenSSH key from {path}")]
    OpenSshParse {
        path: PathBuf,
        #[source]
        source: ssh_key::Error,
    },
    #[error("unsupported key algorithm: only ed25519 is accepted")]
    UnsupportedAlgorithm,
    #[error("private key is encrypted but no passphrase source provided and no TTY available")]
    EncryptedKeyNoPassphrase,
    #[error("failed to decrypt private key (wrong passphrase?)")]
    DecryptFailed,
    #[error("failed to read passphrase from terminal")]
    TtyRead(#[source] std::io::Error),
    #[error("OS random number generator unavailable")]
    Rng,
    #[error("invalid Argon2 parameters")]
    Argon2Params,
    #[error("Argon2 key derivation failed")]
    Argon2Hash,
}

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("container is not signed but verification was requested")]
    ContainerNotSigned,
    #[error(
        "container is signed but no verification key was provided; \
         pass --psk-* / --verify-key, or --insecure-skip-verify to override"
    )]
    KeyRequired,
    #[error("container is signed with {actual} but {expected} key was provided")]
    AlgorithmMismatch {
        expected: AuthAlgorithm,
        actual: AuthAlgorithm,
    },
    #[error("signature verification failed")]
    VerificationFailed,
    #[error("internal invariant: signed container has no auth_prefix recorded")]
    MissingAuthPrefix,
    #[error("internal invariant: classify said verify but no matching auth field found")]
    MissingAuthField,
}
