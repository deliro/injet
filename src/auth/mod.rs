pub mod classify;
pub mod ed25519;
pub mod error;
pub mod passphrase;
pub mod psk;

pub use classify::{classify, AuthAlgorithm, ExtractDecision};
pub use error::{AuthError, KeyError};
