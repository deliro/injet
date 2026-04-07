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

macro_rules! meta_tag_enum {
    ($( $name:ident = $val:expr ),* $(,)?) => {
        #[repr(u8)]
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        pub(crate) enum MetaTag {
            $( $name = $val, )*
        }
        impl From<MetaTag> for u8 {
            fn from(tag: MetaTag) -> Self {
                tag as u8
            }
        }
        impl std::convert::TryFrom<u8> for MetaTag {
            type Error = ();
            fn try_from(value: u8) -> Result<Self, Self::Error> {
                match value {
                    $( $val => Ok(MetaTag::$name), )*
                    _ => Err(()),
                }
            }
        }
    };
}

meta_tag_enum! {
    Size = 1,
    Filename = 2,
    Hash = 3,
    MetaHash = 4,
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
    use std::assert_eq;

    use super::{Meta, MetaError, MAGIC, VERSION_1, VERSION_2};

    #[test]
    fn test_meta_v2_roundtrip() {
        let meta = Meta::make(Some(1231234), Some("hello.zip".to_string()), Some(u32::MAX));
        let meta_bytes = meta.to_bytes();
        println!("meta v2 bytes: {:02x?}", meta_bytes);
        let mut bytes = meta_bytes.into_iter();
        match Meta::read(&mut bytes) {
            Err(_) => panic!("meta wasn't read (v2)"),
            Ok(v) => assert_eq!(v, meta, "received meta differs (v2)"),
        }
        assert_eq!(bytes.next(), None);
    }

    #[test]
    fn test_meta_v1_parsing() {
        // Test version 1 (legacy): manual bytes, only parse
        let mut v1_bytes = Vec::new();
        let signature = (MAGIC << 3) | (VERSION_1 as u16);
        v1_bytes.extend(signature.to_le_bytes());
        v1_bytes.extend(1231234u32.to_le_bytes());
        let filename = b"hello.zip";
        v1_bytes.push(filename.len() as u8);
        v1_bytes.extend(filename);
        let mut v1_iter = v1_bytes.into_iter();
        let meta_v1 = Meta::read(&mut v1_iter).expect("meta v1 should parse");
        assert_eq!(meta_v1.version, VERSION_1);
        assert_eq!(
            meta_v1.fields.iter().find_map(|f| f.as_size()).unwrap(),
            1231234
        );
        assert_eq!(
            meta_v1.fields.iter().find_map(|f| f.as_filename()).unwrap(),
            "hello.zip"
        );
        assert_eq!(v1_iter.next(), None);
    }

    #[test]
    fn test_meta_v2_skips_unknown_tlv_and_keeps_parsing() {
        // Build a v2 meta byte stream:
        //   signature(v2) | Size(4) | UnknownTag=0x7F len=5 + 5 bytes | Filename "x.zip" | end
        let mut bytes = Vec::new();
        let signature = (MAGIC << 3) | (VERSION_2 as u16);
        bytes.extend(signature.to_le_bytes());

        // Size = 1234
        bytes.push(1); // tag Size
        bytes.push(4); // len
        bytes.extend(1234u32.to_le_bytes());

        // Unknown tag 0x7F with 5-byte payload
        bytes.push(0x7F);
        bytes.push(5);
        bytes.extend([0xAA, 0xBB, 0xCC, 0xDD, 0xEE]);

        // Filename "x.zip"
        bytes.push(2); // tag Filename
        bytes.push(5); // len
        bytes.extend(b"x.zip");

        // end marker
        bytes.push(0);
        bytes.push(0);

        let mut iter = bytes.into_iter();
        let meta = Meta::read(&mut iter).expect("meta should parse despite unknown tag");
        assert_eq!(meta.size(), Some(1234), "Size field must survive unknown tag");
        assert_eq!(
            meta.filename(),
            Some("x.zip"),
            "Filename after unknown tag must still be parsed"
        );
        assert_eq!(iter.next(), None, "stream must be fully consumed");
    }

    #[test]
    fn test_meta_v3_roundtrip_with_header_hash() {
        let meta = Meta::make_v3(
            Some(4242),
            Some("hello.bin".to_string()),
            Some(0xDEADBEEF),
        );
        assert_eq!(meta.version, 3);
        let bytes = meta.to_bytes();
        // Compute the expected CRC directly: it covers everything from the start of `bytes`
        // up to (but not including) the MetaHash TLV. The MetaHash TLV at the tail is
        // 6 bytes (tag + len + 4-byte value), and is followed by the 2-byte end marker.
        let meta_hash_tlv_len = 1 /* tag */ + 1 /* len */ + 4 /* u32 */;
        let end_marker_len = 2;
        let covered_end = bytes.len() - end_marker_len - meta_hash_tlv_len;
        let expected_crc = crc32fast::hash(&bytes[..covered_end]);

        let mut iter = bytes.into_iter();
        let parsed = Meta::read(&mut iter).expect("v3 must parse");
        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.size(), Some(4242));
        assert_eq!(parsed.filename(), Some("hello.bin"));
        assert_eq!(parsed.hash(), Some(0xDEADBEEF));
        assert_eq!(
            parsed.meta_hash(),
            Some(expected_crc),
            "MetaHash must equal CRC32 of the preceding header bytes"
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_meta_v3_header_hash_mismatch_detected() {
        let meta = Meta::make_v3(Some(123), Some("a.bin".into()), Some(0));
        let mut bytes = meta.to_bytes();
        // Flip one bit somewhere inside the Filename field (which is BEFORE MetaHash).
        // Layout for v3 with size=123, filename="a.bin" (5 bytes), hash=0:
        //   sig:        2 bytes  (offsets 0..2)
        //   Size TLV:   1+1+4 = 6 bytes  (offsets 2..8)
        //   Filename:   1+1+5 = 7 bytes  (offsets 8..15) — value at 10..15
        // Flip a bit inside the filename value.
        bytes[12] ^= 0x01;
        let mut iter = bytes.into_iter();
        let err = Meta::read(&mut iter).expect_err("must detect header tampering");
        assert!(
            matches!(err, MetaError::HeaderHashMismatch),
            "expected HeaderHashMismatch, got {err:?}"
        );
    }

    #[test]
    fn test_meta_v2_rejects_non_utf8_filename() {
        let mut bytes = Vec::new();
        let signature = (MAGIC << 3) | (VERSION_2 as u16);
        bytes.extend(signature.to_le_bytes());
        bytes.push(2); // Filename tag
        bytes.push(3);
        bytes.extend(&[0xFF, 0xFE, 0xFD]); // not valid UTF-8
        bytes.push(0);
        bytes.push(0);
        let mut iter = bytes.into_iter();
        let err = Meta::read(&mut iter).expect_err("must reject non-UTF-8 filename");
        assert!(matches!(err, MetaError::MalformedFilename));
    }

    #[test]
    fn test_meta_v2_rejects_size_field_with_wrong_length() {
        let mut bytes = Vec::new();
        let signature = (MAGIC << 3) | (VERSION_2 as u16);
        bytes.extend(signature.to_le_bytes());
        bytes.push(1); // Size tag
        bytes.push(3); // wrong: should be 4
        bytes.extend(&[0, 0, 0]);
        bytes.push(0);
        bytes.push(0);
        let mut iter = bytes.into_iter();
        let err = Meta::read(&mut iter).expect_err("malformed Size must error");
        assert!(matches!(err, MetaError::MalformedField), "got {err:?}");
    }
}
