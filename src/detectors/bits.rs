//! Bitstring helpers shared by detectors without error correction, whose
//! payloads are compared to manifest values bit by bit.

use base64::prelude::{BASE64_STANDARD, Engine as _};

pub fn bits_to_bytes(bits: &str) -> Vec<u8> {
    bits.as_bytes()
        .chunks(8)
        .map(|chunk| {
            let byte = chunk.iter().fold(0u8, |acc, &b| (acc << 1) | (b - b'0'));
            // Left-align a trailing partial byte.
            byte << (8 - chunk.len())
        })
        .collect()
}

pub fn bytes_to_bits(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:08b}")).collect()
}

/// The first `nbits` bits of a soft-binding block value, MSB first, or `None`
/// if it doesn't hold that many. Accepts a bitstring, a UUID, hex or base64.
pub fn value_bits(value: &str, nbits: usize) -> Option<String> {
    let value = value.trim();
    let bits = if value.len() >= nbits && value.chars().all(|c| c == '0' || c == '1') {
        value.to_owned()
    } else if let Some(bytes) = uuid_bytes(value).or_else(|| hex::decode(value).ok()) {
        bytes_to_bits(&bytes)
    } else {
        bytes_to_bits(&BASE64_STANDARD.decode(value).ok()?)
    };
    (bits.len() >= nbits).then(|| bits[..nbits].to_owned())
}

fn uuid_bytes(value: &str) -> Option<Vec<u8>> {
    let parts: Vec<&str> = value.split('-').collect();
    let lengths: Vec<usize> = parts.iter().map(|p| p.len()).collect();
    if lengths != [8, 4, 4, 4, 12] {
        return None;
    }
    hex::decode(parts.concat()).ok()
}

/// The share of positions where two equal-length bitstrings agree.
pub fn agreement(a: &str, b: &str) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let same = a.bytes().zip(b.bytes()).filter(|(x, y)| x == y).count();
    same as f32 / a.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_and_bytes_round_trip() {
        let bytes: Vec<u8> = (0..32).map(|i| i * 7).collect();
        assert_eq!(bits_to_bytes(&bytes_to_bits(&bytes)), bytes);
        assert_eq!(bits_to_bytes("1010"), [0b1010_0000]);
    }

    #[test]
    fn parses_value_encodings() {
        let bytes = [0xA5u8; 32];
        let bits = bytes_to_bits(&bytes);
        let b64 = BASE64_STANDARD.encode(bytes);
        assert_eq!(value_bits(&b64, 256).as_deref(), Some(&*bits));
        assert_eq!(
            value_bits(&hex::encode(bytes), 256).as_deref(),
            Some(&*bits)
        );
        assert_eq!(value_bits(&bits, 256).as_deref(), Some(&*bits));
        assert_eq!(value_bits(&b64, 100).as_deref(), Some(&bits[..100]));
        assert_eq!(value_bits("1*0101", 4), None);
        assert_eq!(value_bits(&b64, 512), None);
    }

    #[test]
    fn parses_uuids() {
        let uuid = "a5a5a5a5-a5a5-4a5a-8a5a-a5a5a5a5a5a5";
        let bits = value_bits(uuid, 100).unwrap();
        assert_eq!(&bits[..8], "10100101");
        assert_eq!(&bits[48..52], "0100");
    }

    #[test]
    fn agreement_counts_matching_bits() {
        assert_eq!(agreement("1100", "1000"), 0.75);
        assert_eq!(agreement("1", "10"), 0.0);
    }
}
