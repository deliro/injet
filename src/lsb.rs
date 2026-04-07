use itertools::Itertools;

pub type Seed = String;

#[must_use]
#[inline]
pub fn to_bits(val: u8) -> [u8; 8] {
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

#[inline]
pub fn iter_dots(width: u32, height: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..width).cartesian_product(0..height)
}

fn seed_to_array(seed: &str) -> [u8; 32] {
    *blake3::hash(seed.as_bytes()).as_bytes()
}

pub fn pseudo_shuffle_coords(
    width: u32,
    height: u32,
    seed: &Seed,
) -> impl Iterator<Item = (u32, u32)> {
    let total_pixels = u64::from(width).saturating_mul(u64::from(height));
    // Round total bits up to an even number so Feistel halves are symmetric.
    let bits_min: u32 = if total_pixels <= 1 {
        2
    } else {
        64_u32.saturating_sub(total_pixels.saturating_sub(1).leading_zeros())
    };
    let bits_even = if bits_min.is_multiple_of(2) {
        bits_min
    } else {
        bits_min.saturating_add(1)
    };
    let bits = bits_even.max(2);
    let half = bits / 2;
    let mask = if half >= 64 {
        u64::MAX
    } else {
        (1_u64 << half).saturating_sub(1)
    };
    let total_space = if bits >= 64 { u64::MAX } else { 1_u64 << bits };
    let key = seed_to_array(seed);
    let height_u64 = u64::from(height);

    let round_fn = move |value: u64, round: u8| -> u64 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&key);
        hasher.update(&[round]);
        hasher.update(&value.to_le_bytes());
        let hash = hasher.finalize();
        let hash_bytes = hash.as_bytes();
        let buf: [u8; 8] = hash_bytes
            .get(..8)
            .and_then(|s| s.try_into().ok())
            .unwrap_or_default();
        u64::from_le_bytes(buf) & mask
    };

    let feistel = move |input: u64| -> u64 {
        let mut left = (input >> half) & mask;
        let mut right = input & mask;
        for round in 0_u8..4 {
            let new_left = right;
            let new_right = left ^ round_fn(right, round);
            left = new_left;
            right = new_right;
        }
        (left << half) | right
    };

    (0..total_space)
        .map(feistel)
        .filter(move |value| *value < total_pixels)
        .filter_map(move |value| {
            let x = value
                .checked_div(height_u64)
                .and_then(|v| u32::try_from(v).ok())?;
            let y = value
                .checked_rem(height_u64)
                .and_then(|v| u32::try_from(v).ok())?;
            Some((x, y))
        })
}

pub fn gen_dots(width: u32, height: u32, seed: Option<&Seed>) -> impl Iterator<Item = (u32, u32)> {
    match seed {
        Some(seed) => itertools::Either::Left(pseudo_shuffle_coords(width, height, seed)),
        None => itertools::Either::Right(iter_dots(width, height)),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::pseudo_shuffle_coords;

    #[test]
    fn test_pseudo_shuffle_is_deterministic_for_same_seed() {
        let a: Vec<_> = pseudo_shuffle_coords(20, 20, &"abc".to_string()).collect();
        let b: Vec<_> = pseudo_shuffle_coords(20, 20, &"abc".to_string()).collect();
        assert_eq!(a, b, "same seed must yield identical order");
        let c: Vec<_> = pseudo_shuffle_coords(20, 20, &"abd".to_string()).collect();
        assert_ne!(a, c, "different seed must yield different order");
        // Permutation property: every coordinate appears exactly once.
        let mut sorted = a.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 400);
    }
}
