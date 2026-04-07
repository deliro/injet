use thiserror::Error;

pub const MAGIC: u16 = 0xd2d;
pub const VERSION_1: u8 = 1;
pub const VERSION_2: u8 = 2;
pub const VERSION_3: u8 = 3;

#[inline]
fn to_bits(val: u8) -> [u8; 8] {
    [
        (val >> 7) & 1,
        (val >> 6) & 1,
        (val >> 5) & 1,
        (val >> 4) & 1,
        (val >> 3) & 1,
        (val >> 2) & 1,
        (val >> 1) & 1,
        val & 1,
    ]
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum MetaTag {
    Size = 1,
    Filename = 2,
    Hash = 3,
    MetaHash = 4,
}

impl From<MetaTag> for u8 {
    fn from(tag: MetaTag) -> u8 {
        tag as u8
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
        }
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = vec![u8::from(self.tag())];
        let value = match self {
            MetaField::Size(sz) => sz.to_le_bytes().to_vec(),
            MetaField::Filename(s) => s.as_bytes().to_vec(),
            MetaField::Hash(h) => h.to_le_bytes().to_vec(),
            MetaField::MetaHash(h) => h.to_le_bytes().to_vec(),
        };
        if value.len() > 255 {
            result.push(0x00);
            result.extend((value.len() as u16).to_le_bytes());
        } else {
            result.push(value.len() as u8);
        }
        result.extend(value);
        result
    }
    fn read_tlv_len<T: Iterator<Item = u8>>(
        iter: &mut T,
        first_len_byte: u8,
    ) -> Result<usize, MetaError> {
        if first_len_byte != 0 {
            return Ok(first_len_byte as usize);
        }
        let hi = iter.next().ok_or(MetaError::NoBytes)?;
        let lo = iter.next().ok_or(MetaError::NoBytes)?;
        Ok(u16::from_le_bytes([hi, lo]) as usize)
    }

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
        let tag = match MetaTag::try_from(tag_byte) {
            Ok(t) => t,
            Err(_) => return Ok(MetaFieldParseResult::Skip),
        };
        let field = match tag {
            MetaTag::Size => MetaField::Size(parse_u32_field(bytes)?),
            MetaTag::Filename => MetaField::Filename(parse_filename(bytes)?),
            MetaTag::Hash => MetaField::Hash(parse_u32_field(bytes)?),
            MetaTag::MetaHash => MetaField::MetaHash(parse_u32_field(bytes)?),
        };
        Ok(MetaFieldParseResult::Field(field))
    }
    pub fn as_size(&self) -> Option<u32> {
        if let MetaField::Size(sz) = self {
            Some(*sz)
        } else {
            None
        }
    }
    pub fn as_filename(&self) -> Option<&str> {
        if let MetaField::Filename(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_hash(&self) -> Option<u32> {
        if let MetaField::Hash(h) = self {
            Some(*h)
        } else {
            None
        }
    }
    pub fn as_meta_hash(&self) -> Option<u32> {
        if let MetaField::MetaHash(h) = self {
            Some(*h)
        } else {
            None
        }
    }
    pub fn from_v1_header<T: Iterator<Item = u8>>(
        iter: &mut T,
    ) -> Result<Vec<MetaField>, MetaError> {
        let header_rest: Vec<u8> = iter.take(5).collect();
        if header_rest.len() != 5 {
            return Err(MetaError::NoBytes);
        }
        let size = u32::from_le_bytes(header_rest[0..4].try_into().unwrap());
        let filename_size = match header_rest[4] {
            0xFF => None,
            sz => Some(sz),
        };
        let mut fields = vec![MetaField::Size(size)];
        if let Some(sz) = filename_size {
            let filename_vec = iter.take(sz as usize).collect::<Vec<u8>>();
            if filename_vec.len() as u8 != sz {
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

fn parse_u32_field(bytes: Vec<u8>) -> Result<u32, MetaError> {
    let array: [u8; 4] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| MetaError::MalformedField)?;
    Ok(u32::from_le_bytes(array))
}

#[derive(Debug, Eq, PartialEq)]
pub struct Meta {
    pub version: u8,
    pub fields: Vec<MetaField>,
}

impl Meta {
    fn write_all_fields(&self, buf: &mut Vec<u8>) {
        for field in &self.fields {
            buf.extend(field.to_bytes());
        }
    }

    fn write_fields_excluding_meta_hash(&self, buf: &mut Vec<u8>) {
        for field in &self.fields {
            if field.as_meta_hash().is_some() {
                continue;
            }
            buf.extend(field.to_bytes());
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        // Only v2 and v3 are writable. v1 is read-only (legacy format) — `Meta::read` may
        // return version=1, but no production caller writes such a value back out.
        let signature = match self.version {
            VERSION_2 => (MAGIC << 3) | (VERSION_2 as u16),
            VERSION_3 => (MAGIC << 3) | (VERSION_3 as u16),
            v => unreachable!(
                "Meta::to_bytes called with unsupported writer version {v}; \
                 only VERSION_2 and VERSION_3 are writable"
            ),
        };
        let mut result = Vec::with_capacity(64);
        result.extend(signature.to_le_bytes());
        if self.version == VERSION_3 {
            self.write_fields_excluding_meta_hash(&mut result);
            let crc = crc32fast::hash(&result);
            result.extend(MetaField::MetaHash(crc).to_bytes());
        } else {
            self.write_all_fields(&mut result);
        }
        result.push(0);
        result.push(0);
        result
    }

    pub fn to_bits(&self) -> impl Iterator<Item = u8> {
        self.to_bytes().into_iter().flat_map(to_bits)
    }

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

    pub fn read<T>(value: &mut T) -> Result<Self, MetaError>
    where
        T: Iterator<Item = u8>,
    {
        let sig_bytes: Vec<u8> = value.take(2).collect();
        if sig_bytes.len() != 2 {
            return Err(MetaError::NoBytes);
        }
        let signature = u16::from_le_bytes([sig_bytes[0], sig_bytes[1]]);
        let sign = signature >> 3;
        if sign != MAGIC {
            return Err(MetaError::SignatureMismatch);
        }
        let version = (signature & 0b111) as u8;
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
                        MetaFieldParseResult::Skip => continue,
                    }
                }
                Ok(Meta { version, fields })
            }
            VERSION_3 => {
                // Buffer the v3 header TLV-by-TLV. We need the raw bytes for CRC,
                // and we record the MetaHash offset in the same pass to avoid a
                // re-scan. A byte-by-byte scan for `0x00 0x00` would be unsound
                // because TLV values legitimately contain zero bytes (e.g. a
                // `u32` Size payload).
                let mut body: Vec<u8> = Vec::with_capacity(64);
                body.extend(sig_bytes.iter().copied());
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
                        u16::from_le_bytes([lo, hi]) as usize
                    } else {
                        len_byte as usize
                    };
                    for _ in 0..value_len {
                        let b = value.next().ok_or(MetaError::NoBytes)?;
                        body.push(b);
                    }
                    if tag == u8::from(MetaTag::MetaHash) && meta_hash_offset.is_none() {
                        meta_hash_offset = Some(field_start);
                    }
                }
                let meta_hash_offset = meta_hash_offset.ok_or(MetaError::MetaHashMissing)?;

                // CRC covers everything BEFORE the MetaHash TLV.
                if meta_hash_offset + 6 > body.len() {
                    return Err(MetaError::NoBytes);
                }
                let covered = &body[..meta_hash_offset];
                let expected_crc = u32::from_le_bytes([
                    body[meta_hash_offset + 2],
                    body[meta_hash_offset + 3],
                    body[meta_hash_offset + 4],
                    body[meta_hash_offset + 5],
                ]);
                if crc32fast::hash(covered) != expected_crc {
                    return Err(MetaError::HeaderHashMismatch);
                }

                // Re-parse the verified body via `from_tlv_field`. Re-append the
                // end marker so the End sentinel fires.
                let mut parse_buf: Vec<u8> = body[2..].to_vec();
                parse_buf.push(0);
                parse_buf.push(0);
                let mut iter = parse_buf.into_iter();
                let mut fields = Vec::new();
                loop {
                    match MetaField::from_tlv_field(&mut iter)? {
                        MetaFieldParseResult::Field(field) => fields.push(field),
                        MetaFieldParseResult::End => break,
                        MetaFieldParseResult::Skip => continue,
                    }
                }
                Ok(Meta { version, fields })
            }
            v => Err(MetaError::UnsupportedVersion(v)),
        }
    }

    pub fn size(&self) -> Option<u32> {
        self.fields.iter().find_map(|f| f.as_size())
    }
    pub fn filename(&self) -> Option<&str> {
        self.fields.iter().find_map(|f| f.as_filename())
    }
    pub fn hash(&self) -> Option<u32> {
        self.fields.iter().find_map(|f| f.as_hash())
    }
    pub fn meta_hash(&self) -> Option<u32> {
        self.fields.iter().find_map(|f| f.as_meta_hash())
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
    #[error("Invalid or corrupted filename in metadata")]
    MalformedFilename,
    #[error("Metadata header CRC mismatch")]
    HeaderHashMismatch,
    #[error("Metadata header hash field missing")]
    MetaHashMissing,
    #[error("Malformed metadata field")]
    MalformedField,
}

#[cfg(test)]
mod tests {
    use super::{Meta, MAGIC, VERSION_1, VERSION_2};
    use rstest::rstest;

    fn build_v1_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        let signature = (MAGIC << 3) | (VERSION_1 as u16);
        bytes.extend(signature.to_le_bytes());
        bytes.extend(1231234u32.to_le_bytes());
        let filename = b"hello.zip";
        bytes.push(filename.len() as u8);
        bytes.extend(filename);
        bytes
    }

    fn build_v2_roundtrip() -> Vec<u8> {
        Meta::make(Some(1231234), Some("hello.zip".into()), Some(u32::MAX)).to_bytes()
    }

    fn build_v2_unknown_tlv() -> Vec<u8> {
        let mut bytes = Vec::new();
        let signature = (MAGIC << 3) | (VERSION_2 as u16);
        bytes.extend(signature.to_le_bytes());
        bytes.push(1);
        bytes.push(4);
        bytes.extend(1234u32.to_le_bytes());
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
        Meta::make_v3(Some(4242), Some("hello.bin".into()), Some(0xDEADBEEF)).to_bytes()
    }

    fn build_v3_tampered() -> Vec<u8> {
        let mut bytes = Meta::make_v3(Some(123), Some("a.bin".into()), Some(0)).to_bytes();
        bytes[12] ^= 0x01;
        bytes
    }

    fn build_v2_bad_utf8() -> Vec<u8> {
        let mut bytes = Vec::new();
        let signature = (MAGIC << 3) | (VERSION_2 as u16);
        bytes.extend(signature.to_le_bytes());
        bytes.push(2);
        bytes.push(3);
        bytes.extend(&[0xFF, 0xFE, 0xFD]);
        bytes.push(0);
        bytes.push(0);
        bytes
    }

    fn build_v2_bad_size() -> Vec<u8> {
        let mut bytes = Vec::new();
        let signature = (MAGIC << 3) | (VERSION_2 as u16);
        bytes.extend(signature.to_le_bytes());
        bytes.push(1);
        bytes.push(3);
        bytes.extend(&[0, 0, 0]);
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
    fn meta_parse(#[case] name: &str, #[case] bytes: Vec<u8>) {
        let mut iter = bytes.into_iter();
        insta::assert_debug_snapshot!(name, Meta::read(&mut iter));
    }
}
