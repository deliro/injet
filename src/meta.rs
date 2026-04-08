use thiserror::Error;

pub const MAGIC: u16 = 0xd2d;
pub const VERSION_1: u8 = 1;
pub const VERSION_2: u8 = 2;
pub const VERSION_3: u8 = 3;

const META_HASH_TLV_LEN: usize = 6;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum MetaTag {
    Size,
    Filename,
    Hash,
    MetaHash,
    Mac,
    Signature,
}

impl From<MetaTag> for u8 {
    fn from(tag: MetaTag) -> u8 {
        match tag {
            MetaTag::Size => 1,
            MetaTag::Filename => 2,
            MetaTag::Hash => 3,
            MetaTag::MetaHash => 4,
            MetaTag::Mac => 5,
            MetaTag::Signature => 6,
        }
    }
}

impl TryFrom<u8> for MetaTag {
    type Error = ();
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(MetaTag::Size),
            2 => Ok(MetaTag::Filename),
            3 => Ok(MetaTag::Hash),
            4 => Ok(MetaTag::MetaHash),
            5 => Ok(MetaTag::Mac),
            6 => Ok(MetaTag::Signature),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetaField {
    Size(u32),
    Filename(String),
    Hash(u32),
    MetaHash(u32),
    Mac { salt: [u8; 16], mac: [u8; 32] },
    Signature([u8; 64]),
}

#[derive(Debug)]
pub enum AuthField {
    Mac { salt: [u8; 16], mac: [u8; 32] },
    Signature([u8; 64]),
}

impl AuthField {
    fn into_meta_field(self) -> MetaField {
        match self {
            AuthField::Mac { salt, mac } => MetaField::Mac { salt, mac },
            AuthField::Signature(sig) => MetaField::Signature(sig),
        }
    }
}

pub trait Signer {
    fn sign(&self, auth_input: &[u8]) -> AuthField;
}

pub enum MetaFieldParseResult {
    Field(MetaField),
    End,
    Skip,
}

impl MetaField {
    fn tag(&self) -> MetaTag {
        match self {
            MetaField::Size(_) => MetaTag::Size,
            MetaField::Filename(_) => MetaTag::Filename,
            MetaField::Hash(_) => MetaTag::Hash,
            MetaField::MetaHash(_) => MetaTag::MetaHash,
            MetaField::Mac { .. } => MetaTag::Mac,
            MetaField::Signature(_) => MetaTag::Signature,
        }
    }

    fn value_len(&self) -> usize {
        match self {
            MetaField::Size(_) | MetaField::Hash(_) | MetaField::MetaHash(_) => 4,
            MetaField::Filename(s) => s.len(),
            MetaField::Mac { .. } => 48,
            MetaField::Signature(_) => 64,
        }
    }

    /// Appends the TLV encoding of this field to `out`.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::FieldTooLong`] if the value byte length exceeds [`u16::MAX`].
    pub fn write_into(&self, out: &mut Vec<u8>) -> Result<(), MetaError> {
        let value_len = self.value_len();
        out.push(u8::from(self.tag()));
        if value_len > usize::from(u8::MAX) {
            let len_u16 = u16::try_from(value_len).map_err(|_| MetaError::FieldTooLong)?;
            out.push(0x00);
            out.extend(len_u16.to_le_bytes());
        } else {
            // value_len <= 255 — fits in u8
            let len_u8 = u8::try_from(value_len).map_err(|_| MetaError::FieldTooLong)?;
            out.push(len_u8);
        }
        match self {
            MetaField::Size(sz) => out.extend(sz.to_le_bytes()),
            MetaField::Filename(s) => out.extend(s.as_bytes()),
            MetaField::Hash(h) | MetaField::MetaHash(h) => out.extend(h.to_le_bytes()),
            MetaField::Mac { salt, mac } => {
                out.extend(salt.as_ref());
                out.extend(mac.as_ref());
            }
            MetaField::Signature(sig) => out.extend(sig.as_ref()),
        }
        Ok(())
    }

    /// Encodes a single TLV field into a freshly allocated `Vec<u8>`.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::FieldTooLong`] if the value byte length exceeds [`u16::MAX`].
    pub fn to_bytes(&self) -> Result<Vec<u8>, MetaError> {
        let mut out = Vec::with_capacity(self.value_len().saturating_add(4));
        self.write_into(&mut out)?;
        Ok(out)
    }

    fn read_tlv_len<T: Iterator<Item = u8>>(
        iter: &mut T,
        first_len_byte: u8,
    ) -> Result<usize, MetaError> {
        if first_len_byte != 0 {
            return Ok(usize::from(first_len_byte));
        }
        let lo = iter.next().ok_or(MetaError::NoBytes)?;
        let hi = iter.next().ok_or(MetaError::NoBytes)?;
        Ok(usize::from(u16::from_le_bytes([lo, hi])))
    }

    /// Parses a TLV field from a byte iterator.
    ///
    /// # Errors
    ///
    /// Returns a [`MetaError`] when the iterator is exhausted prematurely or the field
    /// value is malformed (bad UTF-8, wrong length, etc).
    pub fn from_tlv_field<T: Iterator<Item = u8>>(
        iter: &mut T,
    ) -> Result<MetaFieldParseResult, MetaError> {
        let tag_byte = iter.next().ok_or(MetaError::NoBytes)?;
        let len = iter.next().ok_or(MetaError::NoBytes)?;
        if tag_byte == 0 && len == 0 {
            return Ok(MetaFieldParseResult::End);
        }
        let actual_len = Self::read_tlv_len(iter, len)?;
        let bytes: Vec<u8> = iter.take(actual_len).collect();
        if bytes.len() != actual_len {
            return Err(MetaError::NoBytes);
        }
        let Ok(tag) = MetaTag::try_from(tag_byte) else {
            return Ok(MetaFieldParseResult::Skip);
        };
        let field = match tag {
            MetaTag::Size => MetaField::Size(parse_u32_field(&bytes)?),
            MetaTag::Filename => MetaField::Filename(parse_filename(bytes)?),
            MetaTag::Hash => MetaField::Hash(parse_u32_field(&bytes)?),
            MetaTag::MetaHash => MetaField::MetaHash(parse_u32_field(&bytes)?),
            MetaTag::Mac => {
                if bytes.len() != 48 {
                    return Err(MetaError::MalformedField);
                }
                let salt: [u8; 16] = bytes
                    .get(..16)
                    .and_then(|s| <[u8; 16]>::try_from(s).ok())
                    .ok_or(MetaError::MalformedField)?;
                let mac: [u8; 32] = bytes
                    .get(16..48)
                    .and_then(|s| <[u8; 32]>::try_from(s).ok())
                    .ok_or(MetaError::MalformedField)?;
                MetaField::Mac { salt, mac }
            }
            MetaTag::Signature => {
                if bytes.len() != 64 {
                    return Err(MetaError::MalformedField);
                }
                let sig: [u8; 64] = bytes
                    .try_into()
                    .map_err(|_| MetaError::MalformedField)?;
                MetaField::Signature(sig)
            }
        };
        Ok(MetaFieldParseResult::Field(field))
    }

    #[must_use]
    pub fn as_size(&self) -> Option<u32> {
        if let MetaField::Size(sz) = self {
            Some(*sz)
        } else {
            None
        }
    }

    #[must_use]
    pub fn as_filename(&self) -> Option<&str> {
        if let MetaField::Filename(s) = self {
            Some(s)
        } else {
            None
        }
    }

    #[must_use]
    pub fn as_hash(&self) -> Option<u32> {
        if let MetaField::Hash(h) = self {
            Some(*h)
        } else {
            None
        }
    }

    #[must_use]
    pub fn as_meta_hash(&self) -> Option<u32> {
        if let MetaField::MetaHash(h) = self {
            Some(*h)
        } else {
            None
        }
    }

    #[must_use]
    pub fn as_mac(&self) -> Option<(&[u8; 16], &[u8; 32])> {
        if let MetaField::Mac { salt, mac } = self {
            Some((salt, mac))
        } else {
            None
        }
    }

    #[must_use]
    pub fn as_signature(&self) -> Option<&[u8; 64]> {
        if let MetaField::Signature(sig) = self {
            Some(sig)
        } else {
            None
        }
    }

    /// Parses the legacy v1 fixed header (size + optional filename).
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::NoBytes`] when the iterator is exhausted prematurely or
    /// [`MetaError::MalformedFilename`] when the filename payload is invalid UTF-8.
    pub fn from_v1_header<T: Iterator<Item = u8>>(
        iter: &mut T,
    ) -> Result<Vec<MetaField>, MetaError> {
        let mut header = [0_u8; 5];
        for slot in &mut header {
            *slot = iter.next().ok_or(MetaError::NoBytes)?;
        }
        let [s0, s1, s2, s3, fn_size_byte] = header;
        let size = u32::from_le_bytes([s0, s1, s2, s3]);
        let filename_size = match fn_size_byte {
            0xFF => None,
            sz => Some(sz),
        };
        let mut fields = vec![MetaField::Size(size)];
        if let Some(sz) = filename_size {
            let take_n = usize::from(sz);
            let filename_vec: Vec<u8> = iter.take(take_n).collect();
            if filename_vec.len() != take_n {
                return Err(MetaError::MalformedFilename);
            }
            fields.push(MetaField::Filename(parse_filename(filename_vec)?));
        }
        Ok(fields)
    }
}

fn parse_filename(bytes: Vec<u8>) -> Result<String, MetaError> {
    String::from_utf8(bytes).map_err(|_| MetaError::MalformedFilename)
}

fn parse_u32_field(bytes: &[u8]) -> Result<u32, MetaError> {
    let array: [u8; 4] = bytes.try_into().map_err(|_| MetaError::MalformedField)?;
    Ok(u32::from_le_bytes(array))
}

#[derive(Debug, Eq, PartialEq)]
pub struct Meta {
    pub version: u8,
    pub fields: Vec<MetaField>,
}

impl Meta {
    fn write_all_fields(&self, buf: &mut Vec<u8>) -> Result<(), MetaError> {
        for field in &self.fields {
            field.write_into(buf)?;
        }
        Ok(())
    }

    fn write_fields_excluding_meta_hash_and_auth(&self, buf: &mut Vec<u8>) -> Result<(), MetaError> {
        for field in &self.fields {
            match field {
                MetaField::MetaHash(_) | MetaField::Mac { .. } | MetaField::Signature(_) => {}
                _ => field.write_into(buf)?,
            }
        }
        Ok(())
    }

    fn write_auth_fields(&self, buf: &mut Vec<u8>) -> Result<(), MetaError> {
        let mut count: usize = 0;
        for field in &self.fields {
            if matches!(field, MetaField::Mac { .. } | MetaField::Signature(_)) {
                field.write_into(buf)?;
                count = count.saturating_add(1);
            }
        }
        if count > 1 {
            return Err(MetaError::MultipleAuthFields);
        }
        Ok(())
    }

    /// Serializes the metadata header.
    ///
    /// Only versions 2 and 3 are writable. v1 is read-only legacy format —
    /// `Meta::read` may return version=1, but no production caller writes such a value back.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::UnsupportedWriteVersion`] for any version other than 2 or 3,
    /// or [`MetaError::FieldTooLong`] when an embedded field exceeds the TLV length limit.
    pub fn to_bytes(&self) -> Result<Vec<u8>, MetaError> {
        let signature: u16 = match self.version {
            VERSION_2 => (MAGIC << 3_u32) | u16::from(VERSION_2),
            VERSION_3 => (MAGIC << 3_u32) | u16::from(VERSION_3),
            v => return Err(MetaError::UnsupportedWriteVersion(v)),
        };
        let mut result = Vec::with_capacity(64);
        result.extend(signature.to_le_bytes());
        if self.version == VERSION_3 {
            self.write_fields_excluding_meta_hash_and_auth(&mut result)?;
            let crc = crc32fast::hash(&result);
            MetaField::MetaHash(crc).write_into(&mut result)?;
            self.write_auth_fields(&mut result)?;
        } else {
            self.write_all_fields(&mut result)?;
        }
        result.push(0);
        result.push(0);
        Ok(result)
    }

    /// Serialize the metadata header, compute `auth_input = prefix || payload`,
    /// invoke `signer.sign(auth_input)`, and append the resulting auth TLV
    /// followed by the end marker.
    ///
    /// Only valid for `VERSION_3` metadata.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::UnsupportedWriteVersion`] if `self.version != VERSION_3`,
    /// or [`MetaError::FieldTooLong`] / [`MetaError::MultipleAuthFields`] from
    /// the underlying `MetaField::write_into` / format invariant checks.
    pub fn to_bytes_with_auth<S: Signer + ?Sized>(
        &self,
        payload: &[u8],
        signer: &S,
    ) -> Result<Vec<u8>, MetaError> {
        if self.version != VERSION_3 {
            return Err(MetaError::UnsupportedWriteVersion(self.version));
        }
        let mut result = Vec::with_capacity(64);
        let signature: u16 = (MAGIC << 3_u32) | u16::from(VERSION_3);
        result.extend(signature.to_le_bytes());
        self.write_fields_excluding_meta_hash_and_auth(&mut result)?;
        let crc = crc32fast::hash(&result);
        MetaField::MetaHash(crc).write_into(&mut result)?;
        // auth_input = prefix bytes (header up to and including MetaHash TLV) || payload
        let mut auth_input: Vec<u8> = Vec::with_capacity(
            result.len().saturating_add(payload.len()),
        );
        auth_input.extend_from_slice(&result);
        auth_input.extend_from_slice(payload);
        let auth_field = signer.sign(&auth_input);
        auth_field.into_meta_field().write_into(&mut result)?;
        result.push(0);
        result.push(0);
        Ok(result)
    }

    #[must_use]
    pub fn make(size: Option<u32>, filename: Option<String>, hash: Option<u32>) -> Self {
        let mut fields = Vec::new();
        if let Some(size) = size {
            fields.push(MetaField::Size(size));
        }
        if let Some(filename) = filename {
            fields.push(MetaField::Filename(filename));
        }
        if let Some(hash) = hash {
            fields.push(MetaField::Hash(hash));
        }
        Self {
            version: VERSION_2,
            fields,
        }
    }

    #[must_use]
    pub fn make_v3(size: Option<u32>, filename: Option<String>, hash: Option<u32>) -> Self {
        let mut fields = Vec::new();
        if let Some(size) = size {
            fields.push(MetaField::Size(size));
        }
        if let Some(filename) = filename {
            fields.push(MetaField::Filename(filename));
        }
        if let Some(hash) = hash {
            fields.push(MetaField::Hash(hash));
        }
        // MetaHash(0) is a placeholder. `Meta::to_bytes` skips this entry via
        // `write_fields_excluding_meta_hash`, then computes the real CRC over the
        // serialized prefix and appends a fresh MetaHash TLV with the correct value.
        fields.push(MetaField::MetaHash(0));
        Self {
            version: VERSION_3,
            fields,
        }
    }

    /// Reads a metadata header from a byte iterator.
    ///
    /// # Errors
    ///
    /// Returns a [`MetaError`] for malformed signatures, unsupported versions, premature
    /// end-of-stream, header CRC mismatch, or malformed individual fields.
    pub fn read<T>(value: &mut T) -> Result<Self, MetaError>
    where
        T: Iterator<Item = u8>,
    {
        let sig0 = value.next().ok_or(MetaError::NoBytes)?;
        let sig1 = value.next().ok_or(MetaError::NoBytes)?;
        let signature = u16::from_le_bytes([sig0, sig1]);
        let sign = signature >> 3_u32;
        if sign != MAGIC {
            return Err(MetaError::SignatureMismatch);
        }
        let version_low = signature & 0b0000_0111_u16;
        let version = u8::try_from(version_low).map_err(|_| MetaError::MalformedField)?;
        match version {
            VERSION_1 => {
                let fields = MetaField::from_v1_header(value)?;
                Ok(Meta { version, fields })
            }
            VERSION_2 => {
                let mut fields = Vec::new();
                loop {
                    match MetaField::from_tlv_field(value)? {
                        MetaFieldParseResult::Field(field) => fields.push(field),
                        MetaFieldParseResult::End => break,
                        MetaFieldParseResult::Skip => {}
                    }
                }
                Ok(Meta { version, fields })
            }
            VERSION_3 => Self::read_v3(value, [sig0, sig1]),
            v => Err(MetaError::UnsupportedVersion(v)),
        }
    }

    fn read_v3<T>(value: &mut T, sig_bytes: [u8; 2]) -> Result<Self, MetaError>
    where
        T: Iterator<Item = u8>,
    {
        // Buffer the v3 header TLV-by-TLV. We need the raw bytes for CRC,
        // and we record the MetaHash offset in the same pass to avoid a
        // re-scan. A byte-by-byte scan for `0x00 0x00` would be unsound
        // because TLV values legitimately contain zero bytes (e.g. a
        // `u32` Size payload).
        let mut body: Vec<u8> = Vec::with_capacity(64);
        body.extend(sig_bytes);
        let mut meta_hash_offset: Option<usize> = None;
        loop {
            let field_start = body.len();
            let tag = value.next().ok_or(MetaError::NoBytes)?;
            let len_byte = value.next().ok_or(MetaError::NoBytes)?;
            if tag == 0 && len_byte == 0 {
                // End marker — NOT pushed into `body` (CRC scope ends here).
                break;
            }
            body.push(tag);
            body.push(len_byte);
            let value_len = if len_byte == 0 {
                let lo = value.next().ok_or(MetaError::NoBytes)?;
                let hi = value.next().ok_or(MetaError::NoBytes)?;
                body.push(lo);
                body.push(hi);
                usize::from(u16::from_le_bytes([lo, hi]))
            } else {
                usize::from(len_byte)
            };
            for _ in 0..value_len {
                let next_byte = value.next().ok_or(MetaError::NoBytes)?;
                body.push(next_byte);
            }
            if tag == u8::from(MetaTag::MetaHash) && meta_hash_offset.is_none() {
                meta_hash_offset = Some(field_start);
            }
        }
        let meta_hash_offset = meta_hash_offset.ok_or(MetaError::MetaHashMissing)?;

        // CRC covers everything BEFORE the MetaHash TLV.
        let meta_hash_end = meta_hash_offset
            .checked_add(META_HASH_TLV_LEN)
            .ok_or(MetaError::NoBytes)?;
        let covered = body.get(..meta_hash_offset).ok_or(MetaError::NoBytes)?;
        let meta_hash_tlv = body
            .get(meta_hash_offset..meta_hash_end)
            .ok_or(MetaError::NoBytes)?;
        let &[_, _, c0, c1, c2, c3] = meta_hash_tlv else {
            return Err(MetaError::NoBytes);
        };
        let expected_crc = u32::from_le_bytes([c0, c1, c2, c3]);
        if crc32fast::hash(covered) != expected_crc {
            return Err(MetaError::HeaderHashMismatch);
        }

        // Re-parse the verified body via `from_tlv_field`. Re-append the
        // end marker so the End sentinel fires.
        let body_after_sig = body.get(2..).ok_or(MetaError::NoBytes)?;
        let mut iter = body_after_sig.iter().copied().chain([0_u8, 0_u8]);
        let mut fields = Vec::new();
        loop {
            match MetaField::from_tlv_field(&mut iter)? {
                MetaFieldParseResult::Field(field) => fields.push(field),
                MetaFieldParseResult::End => break,
                MetaFieldParseResult::Skip => {}
            }
        }
        Ok(Meta {
            version: VERSION_3,
            fields,
        })
    }

    #[must_use]
    pub fn size(&self) -> Option<u32> {
        self.fields.iter().find_map(MetaField::as_size)
    }

    #[must_use]
    pub fn filename(&self) -> Option<&str> {
        self.fields.iter().find_map(MetaField::as_filename)
    }

    #[must_use]
    pub fn hash(&self) -> Option<u32> {
        self.fields.iter().find_map(MetaField::as_hash)
    }

    #[must_use]
    pub fn meta_hash(&self) -> Option<u32> {
        self.fields.iter().find_map(MetaField::as_meta_hash)
    }
}

#[derive(Debug, Error)]
pub enum MetaError {
    #[error("Insufficient bytes to parse metadata")]
    NoBytes,
    #[error("Invalid metadata signature")]
    SignatureMismatch,
    #[error("Unsupported metadata version: {0}")]
    UnsupportedVersion(u8),
    #[error("Cannot serialize metadata version {0}: writer supports v2 and v3 only")]
    UnsupportedWriteVersion(u8),
    #[error("Invalid or corrupted filename in metadata")]
    MalformedFilename,
    #[error("Metadata header CRC mismatch")]
    HeaderHashMismatch,
    #[error("Metadata header hash field missing")]
    MetaHashMissing,
    #[error("Malformed metadata field")]
    MalformedField,
    #[error("Metadata field value exceeds maximum TLV length")]
    FieldTooLong,
    #[error("Multiple auth fields in container")]
    MultipleAuthFields,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{Meta, MetaField, MAGIC, VERSION_1, VERSION_2, VERSION_3};
    use rstest::rstest;

    fn build_v1_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        let signature = (MAGIC << 3_u32) | u16::from(VERSION_1);
        bytes.extend(signature.to_le_bytes());
        bytes.extend(1_231_234_u32.to_le_bytes());
        let filename = b"hello.zip";
        bytes.push(u8::try_from(filename.len()).unwrap());
        bytes.extend(filename);
        bytes
    }

    fn build_v2_roundtrip() -> Vec<u8> {
        Meta::make(Some(1_231_234), Some("hello.zip".into()), Some(u32::MAX))
            .to_bytes()
            .unwrap()
    }

    fn build_v2_unknown_tlv() -> Vec<u8> {
        let mut bytes = Vec::new();
        let signature = (MAGIC << 3_u32) | u16::from(VERSION_2);
        bytes.extend(signature.to_le_bytes());
        bytes.push(1);
        bytes.push(4);
        bytes.extend(1234_u32.to_le_bytes());
        bytes.push(0x7F);
        bytes.push(5);
        bytes.extend([0xAA, 0xBB, 0xCC, 0xDD, 0xEE]);
        bytes.push(2);
        bytes.push(5);
        bytes.extend(b"x.zip");
        bytes.push(0);
        bytes.push(0);
        bytes
    }

    fn build_v3_roundtrip() -> Vec<u8> {
        Meta::make_v3(Some(4242), Some("hello.bin".into()), Some(0xDEAD_BEEF))
            .to_bytes()
            .unwrap()
    }

    fn build_v3_tampered() -> Vec<u8> {
        let mut bytes = Meta::make_v3(Some(123), Some("a.bin".into()), Some(0))
            .to_bytes()
            .unwrap();
        if let Some(b) = bytes.get_mut(12) {
            *b ^= 0x01;
        }
        bytes
    }

    fn build_v2_bad_utf8() -> Vec<u8> {
        let mut bytes = Vec::new();
        let signature = (MAGIC << 3_u32) | u16::from(VERSION_2);
        bytes.extend(signature.to_le_bytes());
        bytes.push(2);
        bytes.push(3);
        bytes.extend([0xFF, 0xFE, 0xFD]);
        bytes.push(0);
        bytes.push(0);
        bytes
    }

    fn build_v2_bad_size() -> Vec<u8> {
        let mut bytes = Vec::new();
        let signature = (MAGIC << 3_u32) | u16::from(VERSION_2);
        bytes.extend(signature.to_le_bytes());
        bytes.push(1);
        bytes.push(3);
        bytes.extend([0, 0, 0]);
        bytes.push(0);
        bytes.push(0);
        bytes
    }

    fn build_v3_with_mac_roundtrip() -> Vec<u8> {
        let mut m = Meta::make_v3(Some(7), Some("a.bin".into()), Some(0));
        m.fields.push(MetaField::Mac {
            salt: [0x55_u8; 16],
            mac: [0x77_u8; 32],
        });
        m.to_bytes().unwrap()
    }

    fn build_v3_with_sig_roundtrip() -> Vec<u8> {
        let mut m = Meta::make_v3(Some(7), Some("a.bin".into()), Some(0));
        m.fields.push(MetaField::Signature([0xAB_u8; 64]));
        m.to_bytes().unwrap()
    }

    fn build_v3_with_mac_tampered_prefix_byte() -> Vec<u8> {
        let mut bytes = build_v3_with_mac_roundtrip();
        if let Some(b) = bytes.get_mut(8) {
            *b ^= 0x01;
        }
        bytes
    }

    fn build_v3_with_mac_tampered_mac_value_byte() -> Vec<u8> {
        let mut bytes = build_v3_with_mac_roundtrip();
        let len = bytes.len();
        if let Some(idx) = len.checked_sub(4) {
            if let Some(b) = bytes.get_mut(idx) {
                *b ^= 0x01;
            }
        }
        bytes
    }

    fn build_v3_with_mac_tampered_salt_byte() -> Vec<u8> {
        let mut bytes = build_v3_with_mac_roundtrip();
        let pos_opt = bytes.windows(2).position(|w| {
            w.first().copied() == Some(5_u8) && w.get(1).copied() == Some(48_u8)
        });
        if let Some(pos) = pos_opt {
            if let Some(idx) = pos.checked_add(2) {
                if let Some(b) = bytes.get_mut(idx) {
                    *b ^= 0x01;
                }
            }
        }
        bytes
    }

    fn build_v3_signed_read_by_unsigned_path() -> Vec<u8> {
        build_v3_with_mac_roundtrip()
    }

    fn build_v3_unknown_tlv_after_signature() -> Vec<u8> {
        let mut m = Meta::make_v3(Some(7), Some("a.bin".into()), Some(0));
        m.fields.push(MetaField::Signature([0xCD_u8; 64]));
        let mut bytes = m.to_bytes().unwrap();
        if let Some(new_len) = bytes.len().checked_sub(2) {
            bytes.truncate(new_len);
        }
        bytes.push(0x7F);
        bytes.push(3);
        bytes.extend([0xAA_u8, 0xBB, 0xCC]);
        bytes.push(0);
        bytes.push(0);
        bytes
    }

    fn build_v3_signature_truncated_in_value() -> Vec<u8> {
        let mut m = Meta::make_v3(Some(7), Some("a.bin".into()), Some(0));
        m.fields.push(MetaField::Signature([0xCD_u8; 64]));
        let mut bytes = m.to_bytes().unwrap();
        if let Some(new_len) = bytes.len().checked_sub(40) {
            bytes.truncate(new_len);
        }
        bytes
    }

    fn build_v3_mac_wrong_length_short() -> Vec<u8> {
        let mut bytes = Vec::new();
        let signature = (MAGIC << 3_u32) | u16::from(VERSION_3);
        bytes.extend(signature.to_le_bytes());
        bytes.push(1);
        bytes.push(4);
        bytes.extend(7_u32.to_le_bytes());
        bytes.push(5); // Mac tag
        bytes.push(32); // wrong length
        bytes.extend([0_u8; 32]);
        bytes.push(0);
        bytes.push(0);
        bytes
    }

    fn build_v3_mac_wrong_length_long() -> Vec<u8> {
        let mut bytes = Vec::new();
        let signature = (MAGIC << 3_u32) | u16::from(VERSION_3);
        bytes.extend(signature.to_le_bytes());
        bytes.push(1);
        bytes.push(4);
        bytes.extend(7_u32.to_le_bytes());
        bytes.push(5);
        bytes.push(64);
        bytes.extend([0_u8; 64]);
        bytes.push(0);
        bytes.push(0);
        bytes
    }

    fn build_v3_signature_wrong_length() -> Vec<u8> {
        let mut bytes = Vec::new();
        let signature = (MAGIC << 3_u32) | u16::from(VERSION_3);
        bytes.extend(signature.to_le_bytes());
        bytes.push(1);
        bytes.push(4);
        bytes.extend(7_u32.to_le_bytes());
        bytes.push(6);
        bytes.push(32);
        bytes.extend([0_u8; 32]);
        bytes.push(0);
        bytes.push(0);
        bytes
    }

    #[rstest]
    #[case::v1("v1", build_v1_bytes())]
    #[case::v2_roundtrip("v2_roundtrip", build_v2_roundtrip())]
    #[case::v2_unknown_tlv("v2_unknown_tlv", build_v2_unknown_tlv())]
    #[case::v3_roundtrip("v3_roundtrip", build_v3_roundtrip())]
    #[case::v3_tampered("v3_tampered", build_v3_tampered())]
    #[case::v2_bad_utf8("v2_bad_utf8", build_v2_bad_utf8())]
    #[case::v2_bad_size("v2_bad_size", build_v2_bad_size())]
    #[case::v3_with_mac_roundtrip("v3_with_mac_roundtrip", build_v3_with_mac_roundtrip())]
    #[case::v3_with_sig_roundtrip("v3_with_sig_roundtrip", build_v3_with_sig_roundtrip())]
    #[case::v3_with_mac_tampered_prefix_byte("v3_with_mac_tampered_prefix_byte", build_v3_with_mac_tampered_prefix_byte())]
    #[case::v3_with_mac_tampered_mac_value_byte("v3_with_mac_tampered_mac_value_byte", build_v3_with_mac_tampered_mac_value_byte())]
    #[case::v3_with_mac_tampered_salt_byte("v3_with_mac_tampered_salt_byte", build_v3_with_mac_tampered_salt_byte())]
    #[case::v3_signed_read_by_unsigned_path("v3_signed_read_by_unsigned_path", build_v3_signed_read_by_unsigned_path())]
    #[case::v3_unknown_tlv_after_signature("v3_unknown_tlv_after_signature", build_v3_unknown_tlv_after_signature())]
    #[case::v3_signature_truncated_in_value("v3_signature_truncated_in_value", build_v3_signature_truncated_in_value())]
    #[case::v3_mac_wrong_length_short("v3_mac_wrong_length_short", build_v3_mac_wrong_length_short())]
    #[case::v3_mac_wrong_length_long("v3_mac_wrong_length_long", build_v3_mac_wrong_length_long())]
    #[case::v3_signature_wrong_length("v3_signature_wrong_length", build_v3_signature_wrong_length())]
    fn meta_parse(#[case] name: &str, #[case] bytes: Vec<u8>) {
        let mut iter = bytes.into_iter();
        insta::assert_debug_snapshot!(name, Meta::read(&mut iter));
    }

    #[test]
    fn to_bytes_with_auth_round_trip_mac() {
        struct Fake;
        impl super::Signer for Fake {
            fn sign(&self, auth_input: &[u8]) -> super::AuthField {
                let mut salt = [0_u8; 16];
                let mut mac = [0_u8; 32];
                for (slot, byte) in salt.iter_mut().zip(auth_input.iter().take(16)) {
                    *slot = *byte;
                }
                for (slot, byte) in mac.iter_mut().zip(auth_input.iter().take(32)) {
                    *slot = *byte;
                }
                super::AuthField::Mac { salt, mac }
            }
        }
        let meta = super::Meta::make_v3(Some(8), Some("x.bin".into()), Some(0));
        let payload = b"PAYLOAD!";
        let bytes = meta.to_bytes_with_auth(payload, &Fake).unwrap();
        let mut iter = bytes.into_iter();
        let parsed = super::Meta::read(&mut iter).unwrap();
        let mac_field = parsed.fields.iter().find_map(|f| match f {
            super::MetaField::Mac { salt, mac } => Some((*salt, *mac)),
            _ => None,
        });
        assert!(mac_field.is_some(), "Mac field should be present after round-trip");
    }

    #[test]
    fn to_bytes_with_auth_round_trip_signature() {
        struct Fake;
        impl super::Signer for Fake {
            fn sign(&self, _auth_input: &[u8]) -> super::AuthField {
                super::AuthField::Signature([0x99_u8; 64])
            }
        }
        let meta = super::Meta::make_v3(Some(4), Some("y.bin".into()), Some(0));
        let payload = b"DATA";
        let bytes = meta.to_bytes_with_auth(payload, &Fake).unwrap();
        let mut iter = bytes.into_iter();
        let parsed = super::Meta::read(&mut iter).unwrap();
        let sig = parsed.fields.iter().find_map(|f| match f {
            super::MetaField::Signature(s) => Some(*s),
            _ => None,
        });
        assert_eq!(sig, Some([0x99_u8; 64]));
    }

    #[test]
    fn to_bytes_with_auth_rejects_non_v3() {
        struct Fake;
        impl super::Signer for Fake {
            fn sign(&self, _auth_input: &[u8]) -> super::AuthField {
                super::AuthField::Signature([0_u8; 64])
            }
        }
        let mut meta = super::Meta::make(Some(1), Some("z.bin".into()), Some(0));
        // make() returns VERSION_2; the method should refuse to write a v2.
        meta.version = super::VERSION_2;
        let err = meta.to_bytes_with_auth(b"x", &Fake).unwrap_err();
        assert!(matches!(err, super::MetaError::UnsupportedWriteVersion(2)));
    }
}
