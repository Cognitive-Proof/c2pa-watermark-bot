//! End-to-end tests through the HTTP router. They need the TrustMark models in
//! `models/` (run `scripts/fetch-models.sh`) and are skipped without them.

use std::{io::Cursor, path::Path, sync::Arc};

use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use c2pa_watermark_bot::{
    api, detectors::trustmark::TrustmarkDetector, load_state, remote::RemoteConfig,
};
use http_body_util::BodyExt as _;
use image::{DynamicImage, ImageFormat, RgbImage};
use serde_json::Value;
use tower::ServiceExt as _;
use trustmark::Variant;

const MODELS: &str = "models";
const BITS: &str = "0110100001101001";

fn models_present() -> bool {
    let present = [Variant::Q, Variant::C, Variant::P]
        .iter()
        .all(|&v| TrustmarkDetector::models_present(Path::new(MODELS), v));
    if !present {
        eprintln!("skipping: TrustMark models not found in {MODELS}/");
    }
    present
}

/// A smooth, photo-like test image. Flat images give the decoder little to work with.
fn test_image() -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(512, 384, |x, y| {
        let r = (x * 255 / 512) as u8;
        let g = (y * 255 / 384) as u8;
        let b = ((x + y) % 256) as u8;
        image::Rgb([r, g, b])
    }))
}

fn png_bytes(img: &DynamicImage) -> Vec<u8> {
    let mut out = Vec::new();
    img.write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
        .unwrap();
    out
}

async fn post_detect(app: axum::Router, body: Vec<u8>, content_type: &str) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::post("/v1/detect")
                .header(header::CONTENT_TYPE, content_type)
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap())
}

fn app() -> axum::Router {
    let state = load_state(Path::new(MODELS), RemoteConfig::default()).unwrap();
    api::router(Arc::new(state), 25 * 1024 * 1024)
}

#[tokio::test]
async fn finds_trustmark_q_watermark() {
    if !models_present() {
        return;
    }
    let encoder = TrustmarkDetector::load(Path::new(MODELS), Variant::Q).unwrap();
    let marked = DynamicImage::ImageRgb8(
        encoder
            .encode(BITS.to_owned(), test_image())
            .unwrap()
            .to_rgb8(),
    );

    let (status, body) = post_detect(app(), png_bytes(&marked), "image/png").await;
    assert_eq!(status, StatusCode::OK, "{body:#}");

    let found: Vec<&str> = body["found"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(found.contains(&"com.adobe.trustmark.Q"), "{body:#}");

    let q = body["watermarks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|w| w["alg"] == "com.adobe.trustmark.Q")
        .unwrap();
    assert!(
        q["payload"]["bits"].as_str().unwrap().starts_with(BITS),
        "{body:#}"
    );
    assert_eq!(body["c2pa"]["present"], false);
}

#[tokio::test]
async fn clean_image_has_no_watermark() {
    if !models_present() {
        return;
    }
    let (status, body) = post_detect(app(), png_bytes(&test_image()), "image/png").await;
    assert_eq!(status, StatusCode::OK, "{body:#}");
    assert_eq!(body["found"], serde_json::json!([]), "{body:#}");

    // Algorithms without a public decoder are reported, not silently skipped.
    let not_checked = body["not_checked"].as_array().unwrap();
    assert!(
        not_checked
            .iter()
            .any(|a| a["alg"] == "com.digimarc.validate.1")
    );
    assert!(
        not_checked
            .iter()
            .any(|a| a["alg"] == "com.adobe.trustmark.F")
    );
}

#[tokio::test]
async fn accepts_multipart_upload() {
    if !models_present() {
        return;
    }
    let boundary = "x-boundary";
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n"
    )
    .into_bytes();
    body.extend(png_bytes(&test_image()));
    body.extend(format!("\r\n--{boundary}--\r\n").into_bytes());

    let (status, body) = post_detect(
        app(),
        body,
        &format!("multipart/form-data; boundary={boundary}"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body:#}");
    assert_eq!(body["media_type"], "image/png");
}

#[tokio::test]
async fn rejects_non_image() {
    let state = load_state(Path::new("does-not-exist"), RemoteConfig::default()).unwrap();
    let app = api::router(Arc::new(state), 1024);
    let (status, _) = post_detect(app, b"not an image".to_vec(), "text/plain").await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

const VIDEOSEAL: &str = "me.reconize.videoseal.1";
const PIXELSEAL: &str = "com.aiwatermark.pixelseal.1";
const INVISMARK: &str = "com.microsoft.invismark.1";

fn onnx_models_present() -> bool {
    let present = [
        "videoseal_detector.onnx",
        "pixelseal_detector.onnx",
        "invismark_decoder.onnx",
    ]
    .iter()
    .all(|f| Path::new(MODELS).join(f).is_file());
    if !present {
        eprintln!("skipping: VideoSeal or InvisMark ONNX models not found in {MODELS}/");
    }
    present
}

fn watermark<'a>(body: &'a Value, alg: &str) -> &'a Value {
    body["watermarks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|w| w["alg"] == alg)
        .unwrap_or_else(|| panic!("no result for {alg}: {body:#}"))
}

/// Checks a fixture written by `scripts/calibrate_videoseal.py` or
/// `scripts/calibrate_invismark.py`: found by its own detector with at least
/// `min_agreement` of the embedded bits, not found by the others, and scored
/// like the Python reference.
async fn check_fixture(name: &str, alg: &str, min_agreement: f64) {
    if !onnx_models_present() {
        return;
    }
    let fixtures = Path::new("tests/fixtures");
    let png = std::fs::read(fixtures.join(format!("{name}_marked.png"))).unwrap();
    let expected: Value = serde_json::from_slice(
        &std::fs::read(fixtures.join(format!("{name}_marked.json"))).unwrap(),
    )
    .unwrap();

    let (status, body) = post_detect(app(), png, "image/png").await;
    assert_eq!(status, StatusCode::OK, "{body:#}");

    let result = watermark(&body, alg);
    assert_eq!(result["found"], true, "{body:#}");
    let decoded = result["payload"]["bits"].as_str().unwrap();
    let embedded = expected["embedded_bits"].as_str().unwrap();
    assert_eq!(decoded.len(), embedded.len());
    let agree = decoded
        .chars()
        .zip(embedded.chars())
        .filter(|(a, b)| a == b)
        .count() as f64
        / embedded.len() as f64;
    assert!(agree >= min_agreement, "{agree} of bits match: {body:#}");

    let score = result["score"].as_f64().unwrap();
    let reference = expected["score_on_fixture"].as_f64().unwrap();
    assert!(
        (score - reference).abs() / reference < 0.15,
        "Rust score {score} vs Python {reference}"
    );

    for other in [VIDEOSEAL, PIXELSEAL, INVISMARK] {
        if other != alg {
            assert_eq!(watermark(&body, other)["found"], false, "{other}: {body:#}");
        }
    }
    let not_checked = body["not_checked"].as_array().unwrap();
    assert!(!not_checked.iter().any(|a| a["alg"] == alg));
}

#[tokio::test]
async fn finds_videoseal_watermark() {
    check_fixture("videoseal", VIDEOSEAL, 0.94).await;
}

#[tokio::test]
async fn finds_pixelseal_watermark() {
    check_fixture("pixelseal", PIXELSEAL, 0.94).await;
}

#[tokio::test]
async fn finds_invismark_watermark() {
    check_fixture("invismark", INVISMARK, 0.94).await;
}

#[tokio::test]
async fn clean_image_scores_below_onnx_thresholds() {
    if !onnx_models_present() {
        return;
    }
    let (_, body) = post_detect(app(), png_bytes(&test_image()), "image/png").await;
    for (alg, max_score) in [(VIDEOSEAL, 2.0), (PIXELSEAL, 2.0), (INVISMARK, 0.66)] {
        let result = watermark(&body, alg);
        assert_eq!(result["found"], false, "{body:#}");
        assert!(result["score"].as_f64().unwrap() < max_score, "{body:#}");
    }
}
