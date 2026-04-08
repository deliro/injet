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
    match (container_auth, user_key, insecure_skip) {
        (None, None, _) => ExtractDecision::Unsigned,
        (None, Some(_), _) => ExtractDecision::RejectUnsignedKeyProvided,
        (Some(_), _, true) => ExtractDecision::SkipVerifyInsecure,
        (Some(_), None, false) => ExtractDecision::RejectSignedNoKey,
        (Some(c), Some(u), false) if c != u => ExtractDecision::RejectAlgorithmMismatch {
            expected: c,
            actual: u,
        },
        (Some(_), Some(_), false) => ExtractDecision::Verify,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{classify, AuthAlgorithm};
    use rstest::rstest;

    #[rstest]
    #[case::unsigned_no_key_no_skip(None, None, false)]
    #[case::unsigned_no_key_skip(None, None, true)]
    #[case::unsigned_psk_key(None, Some(AuthAlgorithm::Psk), false)]
    #[case::unsigned_psk_key_skip(None, Some(AuthAlgorithm::Psk), true)]
    #[case::unsigned_ed_key(None, Some(AuthAlgorithm::Ed25519), false)]
    #[case::unsigned_ed_key_skip(None, Some(AuthAlgorithm::Ed25519), true)]
    #[case::psk_no_key_no_skip(Some(AuthAlgorithm::Psk), None, false)]
    #[case::psk_no_key_skip(Some(AuthAlgorithm::Psk), None, true)]
    #[case::psk_psk_key(Some(AuthAlgorithm::Psk), Some(AuthAlgorithm::Psk), false)]
    #[case::psk_psk_key_skip(Some(AuthAlgorithm::Psk), Some(AuthAlgorithm::Psk), true)]
    #[case::psk_ed_key(Some(AuthAlgorithm::Psk), Some(AuthAlgorithm::Ed25519), false)]
    #[case::psk_ed_key_skip(Some(AuthAlgorithm::Psk), Some(AuthAlgorithm::Ed25519), true)]
    #[case::ed_no_key(Some(AuthAlgorithm::Ed25519), None, false)]
    #[case::ed_no_key_skip(Some(AuthAlgorithm::Ed25519), None, true)]
    #[case::ed_ed_key(Some(AuthAlgorithm::Ed25519), Some(AuthAlgorithm::Ed25519), false)]
    #[case::ed_ed_key_skip(Some(AuthAlgorithm::Ed25519), Some(AuthAlgorithm::Ed25519), true)]
    #[case::ed_psk_key(Some(AuthAlgorithm::Ed25519), Some(AuthAlgorithm::Psk), false)]
    #[case::ed_psk_key_skip(Some(AuthAlgorithm::Ed25519), Some(AuthAlgorithm::Psk), true)]
    fn classify_matrix(
        #[case] container: Option<AuthAlgorithm>,
        #[case] user: Option<AuthAlgorithm>,
        #[case] skip: bool,
    ) {
        let decision = classify(container, user, skip);
        let name = format!("container={container:?}__user={user:?}__skip={skip}");
        insta::assert_debug_snapshot!(name, decision);
    }
}
