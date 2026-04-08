use std::fmt;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AuthAlgorithm {
    Psk,
    Ed25519,
}

impl fmt::Display for AuthAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuthAlgorithm::Psk => f.write_str("psk"),
            AuthAlgorithm::Ed25519 => f.write_str("ed25519"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ExtractDecision {
    Unsigned,
    RejectUnsignedKeyProvided,
    RejectSignedNoKey,
    RejectAlgorithmMismatch {
        expected: AuthAlgorithm,
        actual: AuthAlgorithm,
    },
    Verify,
    SkipVerifyInsecure,
}

#[must_use]
pub fn classify(
    container_auth: Option<AuthAlgorithm>,
    user_key: Option<AuthAlgorithm>,
    insecure_skip: bool,
) -> ExtractDecision {
    // implemented in Task 2
    let _ = (container_auth, user_key, insecure_skip);
    ExtractDecision::Unsigned
}
