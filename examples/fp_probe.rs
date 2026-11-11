//! Measures TrustMark false positives on unmarked images, and whether a
//! second decode of a slightly resized copy separates real watermarks from noise.
//!
//!     cargo run --release --example fp_probe -- <image>...

use std::path::Path;

use image::{DynamicImage, imageops::FilterType};
use trustmark::{Trustmark, Variant, Version};

const PAYLOAD: &str = "10110011100011110000111110000011111100000011111110000000111111110000";

fn perturb(img: &DynamicImage) -> DynamicImage {
    img.resize(
        img.width() * 9 / 10,
        img.height() * 9 / 10,
        FilterType::Lanczos3,
    )
}

fn decode(tm: &Trustmark, img: &DynamicImage) -> Option<String> {
    tm.decode(img.clone()).ok()
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    let variants = [Variant::Q, Variant::C, Variant::P];
    let models: Vec<Trustmark> = variants
        .iter()
        .map(|&v| Trustmark::new(Path::new("models"), v, Version::Bch4).unwrap())
        .collect();

    let mut clean_hits = [0; 3];
    let mut clean_hits_confirmed = [0; 3];
    let mut marked_hits = [0; 3];
    let mut marked_hits_confirmed = [0; 3];

    for path in &paths {
        let img = image::open(path).unwrap();
        let perturbed = perturb(&img);
        for (i, tm) in models.iter().enumerate() {
            if let Some(bits) = decode(tm, &img) {
                clean_hits[i] += 1;
                let again = decode(tm, &perturbed);
                let confirmed = again.as_deref() == Some(bits.as_str());
                clean_hits_confirmed[i] += confirmed as usize;
                println!(
                    "FP  {:?} {path} len={} confirmed={confirmed}",
                    variants[i],
                    bits.len()
                );
            }

            let marked = tm.encode(PAYLOAD.to_owned(), img.clone(), 0.95).unwrap();
            let marked = DynamicImage::ImageRgb8(marked.to_rgb8());
            if decode(tm, &marked).as_deref() == Some(PAYLOAD) {
                marked_hits[i] += 1;
                if decode(tm, &perturb(&marked)).as_deref() == Some(PAYLOAD) {
                    marked_hits_confirmed[i] += 1;
                }
            }
        }
    }

    let n = paths.len();
    for (i, v) in variants.iter().enumerate() {
        println!(
            "{v:?}: clean false positives {}/{n} (confirmed by resize {}), marked found {}/{n} (confirmed by resize {})",
            clean_hits[i], clean_hits_confirmed[i], marked_hits[i], marked_hits_confirmed[i]
        );
    }
}
