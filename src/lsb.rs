use itertools::Itertools;

pub type Seed = String;

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
pub fn iter_dots(w: u32, h: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..w).cartesian_product(0..h)
}

fn seed_to_array(seed: &str) -> [u8; 32] {
    *blake3::hash(seed.as_bytes()).as_bytes()
}

pub fn pseudo_shuffle_coords(w: u32, h: u32, seed: &Seed) -> impl Iterator<Item = (u32, u32)> {
    let n = u64::from(w) * u64::from(h);
    // Round total bits up to an even number so Feistel halves are symmetric.
    let bits_min = if n <= 1 { 2 } else { 64 - (n - 1).leading_zeros() };
    let bits_even = if bits_min % 2 == 0 {
        bits_min
    } else {
        bits_min + 1
    };
    let bits = bits_even.max(2);
    let half = bits / 2;
    let mask = if half >= 64 { u64::MAX } else { (1u64 << half) - 1 };
    let total = if bits >= 64 { u64::MAX } else { 1u64 << bits };
    let key = seed_to_array(seed);

    let round_fn = move |x: u64, r: u32| -> u64 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&key);
        hasher.update(&[r as u8]);
        hasher.update(&x.to_le_bytes());
        let bytes = hasher.finalize();
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes.as_bytes()[..8]);
        u64::from_le_bytes(buf) & mask
    };

    let feistel = move |x: u64| -> u64 {
        let mut l = (x >> half) & mask;
        let mut r = x & mask;
        for round in 0..4u32 {
            let new_l = r;
            let new_r = l ^ round_fn(r, round);
            l = new_l;
            r = new_r;
        }
        (l << half) | r
    };

    (0..total)
        .map(feistel)
        .filter(move |v| *v < n)
        .map(move |v| {
            let x = (v / u64::from(h)) as u32;
            let y = (v % u64::from(h)) as u32;
            (x, y)
        })
}

pub fn gen_dots(w: u32, h: u32, seed: Option<&Seed>) -> impl Iterator<Item = (u32, u32)> {
    match seed {
        Some(seed) => itertools::Either::Left(pseudo_shuffle_coords(w, h, seed)),
        None => itertools::Either::Right(iter_dots(w, h)),
    }
}

#[cfg(test)]
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
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 400);
    }
}
