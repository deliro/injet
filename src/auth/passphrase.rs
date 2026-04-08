use std::path::{Path, PathBuf};

use zeroize::Zeroizing;

use crate::auth::error::KeyError;

const MAX_PASSPHRASE_BYTES: usize = 4096;

#[derive(Debug, Clone)]
pub enum PassphraseSource {
    File(PathBuf),
    Env(String),
    Prompt,
}

/// Load a passphrase from the given source into a zeroizing buffer.
///
/// # Errors
///
/// Propagates I/O, missing-env, empty, oversize, and TTY errors as
/// [`KeyError`] variants.
pub fn load(source: &PassphraseSource) -> Result<Zeroizing<Vec<u8>>, KeyError> {
    match source {
        PassphraseSource::File(path) => load_from_file(path),
        PassphraseSource::Env(name) => load_from_env(name),
        PassphraseSource::Prompt => prompt_tty(),
    }
}

fn load_from_file(path: &Path) -> Result<Zeroizing<Vec<u8>>, KeyError> {
    let bytes = std::fs::read(path).map_err(|source| KeyError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if bytes.len() > MAX_PASSPHRASE_BYTES {
        return Err(KeyError::PassphraseTooLong);
    }
    let trimmed = strip_trailing_newline(bytes);
    if trimmed.is_empty() {
        return Err(KeyError::EmptyPassphrase);
    }
    Ok(Zeroizing::new(trimmed))
}

fn load_from_env(name: &str) -> Result<Zeroizing<Vec<u8>>, KeyError> {
    let value = std::env::var(name).map_err(|_| KeyError::EnvVarMissing(name.to_owned()))?;
    if value.is_empty() {
        return Err(KeyError::EmptyPassphrase);
    }
    if value.len() > MAX_PASSPHRASE_BYTES {
        return Err(KeyError::PassphraseTooLong);
    }
    Ok(Zeroizing::new(value.into_bytes()))
}

fn prompt_tty() -> Result<Zeroizing<Vec<u8>>, KeyError> {
    let value = rpassword::prompt_password("Passphrase: ").map_err(KeyError::TtyRead)?;
    if value.is_empty() {
        return Err(KeyError::EmptyPassphrase);
    }
    if value.len() > MAX_PASSPHRASE_BYTES {
        return Err(KeyError::PassphraseTooLong);
    }
    Ok(Zeroizing::new(value.into_bytes()))
}

fn strip_trailing_newline(mut bytes: Vec<u8>) -> Vec<u8> {
    if bytes.last().copied() == Some(b'\n') {
        bytes.pop();
        if bytes.last().copied() == Some(b'\r') {
            bytes.pop();
        }
    }
    bytes
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{load, PassphraseSource};
    use crate::auth::error::KeyError;
    use std::io::Write;

    fn write_temp(content: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content).unwrap();
        f
    }

    #[test]
    fn load_file_strips_single_lf() {
        let f = write_temp(b"hunter2\n");
        let p = load(&PassphraseSource::File(f.path().to_path_buf())).unwrap();
        assert_eq!(&**p, b"hunter2");
    }

    #[test]
    fn load_file_strips_crlf() {
        let f = write_temp(b"hunter2\r\n");
        let p = load(&PassphraseSource::File(f.path().to_path_buf())).unwrap();
        assert_eq!(&**p, b"hunter2");
    }

    #[test]
    fn load_file_preserves_interior_spaces() {
        let f = write_temp(b"two words\n");
        let p = load(&PassphraseSource::File(f.path().to_path_buf())).unwrap();
        assert_eq!(&**p, b"two words");
    }

    #[test]
    fn load_file_empty_rejected() {
        let f = write_temp(b"");
        let err = load(&PassphraseSource::File(f.path().to_path_buf())).unwrap_err();
        assert!(matches!(err, KeyError::EmptyPassphrase));
    }

    #[test]
    fn load_file_only_newline_rejected() {
        let f = write_temp(b"\n");
        let err = load(&PassphraseSource::File(f.path().to_path_buf())).unwrap_err();
        assert!(matches!(err, KeyError::EmptyPassphrase));
    }

    #[test]
    fn load_file_too_large_rejected() {
        let f = write_temp(&vec![b'A'; 4097]);
        let err = load(&PassphraseSource::File(f.path().to_path_buf())).unwrap_err();
        assert!(matches!(err, KeyError::PassphraseTooLong));
    }

    #[test]
    fn load_env_happy() {
        let name = "INJET_TEST_PSK_HAPPY";
        std::env::set_var(name, "secret");
        let p = load(&PassphraseSource::Env(name.into())).unwrap();
        assert_eq!(&**p, b"secret");
        std::env::remove_var(name);
    }

    #[test]
    fn load_env_missing() {
        let name = "INJET_TEST_PSK_DOES_NOT_EXIST";
        std::env::remove_var(name);
        let err = load(&PassphraseSource::Env(name.into())).unwrap_err();
        assert!(matches!(err, KeyError::EnvVarMissing(_)));
    }

    #[test]
    fn load_env_empty() {
        let name = "INJET_TEST_PSK_EMPTY";
        std::env::set_var(name, "");
        let err = load(&PassphraseSource::Env(name.into())).unwrap_err();
        assert!(matches!(err, KeyError::EmptyPassphrase));
        std::env::remove_var(name);
    }
}
