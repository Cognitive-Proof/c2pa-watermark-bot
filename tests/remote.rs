//! Resolution API queries against a local mock service. Needs the PixelSeal
//! ONNX model in `models/` and is skipped without it.

use std::{path::Path, sync::Arc};

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::Query,
    http::{HeaderMap, Request, StatusCode},
    routing::get,
};
use base64::prelude::{BASE64_STANDARD, Engine as _};
use c2pa_watermark_bot::{
    api, load_state,
    remote::{self, RemoteConfig, Service},
};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use tower::ServiceExt as _;

const PIXELSEAL: &str = "com.aiwatermark.pixelseal.1";

/// A resolution API that matches one binding value and any upload.
async fn mock_service(expected_value: String) -> String {
    let lookup = move |Query(q): Query<std::collections::HashMap<String, String>>| {
        let expected = expected_value.clone();
        async move {
            let matched = q.get("alg").map(String::as_str) == Some(PIXELSEAL)
                && q.get("value") == Some(&expected);
            Json(if matched {
                json!({ "matches": [{ "manifestId": "urn:mock:lookup", "similarityScore": 100 }] })
            } else {
                json!({ "matches": [] })
            })
        }
    };
    let upload = |headers: HeaderMap, body: Bytes| async move {
        let png = headers.get("content-type").and_then(|v| v.to_str().ok()) == Some("image/png");
        if !png || body.is_empty() {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "expected the image as the raw body" })),
            );
        }
        (
            StatusCode::OK,
            Json(
                json!({ "matches": [{ "manifestId": "urn:mock:upload", "endpoint": "https://mock" }] }),
            ),
        )
    };
    let app = Router::new()
        .route("/matches/byBinding", get(lookup))
        .route("/matches/byContent", axum::routing::post(upload));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn looks_up_decoded_payload_and_uploads() {
    if !Path::new("models/pixelseal_detector.onnx").is_file() {
        eprintln!("skipping: PixelSeal model not found in models/");
        return;
    }
    let png = std::fs::read("tests/fixtures/pixelseal_marked.png").unwrap();

    // Decode once without remote queries to learn the payload to expect.
    let offline = api::router(
        Arc::new(load_state(Path::new("models"), RemoteConfig::default()).unwrap()),
        25 << 20,
    );
    let body = post(offline, png.clone()).await;
    let hex = body["watermarks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|w| w["alg"] == PIXELSEAL)
        .unwrap()["payload"]["hex"]
        .as_str()
        .unwrap()
        .to_owned();
    let expected_value = BASE64_STANDARD.encode(hex::decode(hex).unwrap());

    // The upload must be the raw image with its media type, as the spec defines.
    let base_url = mock_service(expected_value.clone()).await;
    let mut state = load_state(Path::new("models"), RemoteConfig::default()).unwrap();
    state.remote = remote::Client::with_services(
        vec![Service {
            name: "aiwatermark".to_owned(),
            base_url,
            algs: vec![PIXELSEAL.to_owned()],
        }],
        RemoteConfig {
            lookups: true,
            uploads: vec!["aiwatermark".to_owned()],
        },
    )
    .unwrap();
    let body = post(api::router(Arc::new(state), 25 << 20), png).await;

    assert_eq!(body["remote_matches"], json!([PIXELSEAL]), "{body:#}");
    let queries = body["remote"]["queries"].as_array().unwrap();
    let lookup = queries.iter().find(|q| q["method"] == "byBinding").unwrap();
    assert_eq!(lookup["value"], expected_value, "{body:#}");
    assert_eq!(lookup["matches"][0]["manifestId"], "urn:mock:lookup");
    let upload = queries.iter().find(|q| q["method"] == "byContent").unwrap();
    assert_eq!(
        upload["matches"][0]["manifestId"], "urn:mock:upload",
        "{body:#}"
    );
}

#[tokio::test]
async fn remote_queries_are_off_by_default() {
    let state = load_state(Path::new("does-not-exist"), RemoteConfig::default()).unwrap();
    let mut png = Vec::new();
    image::DynamicImage::new_rgb8(64, 64)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    let body = post(api::router(Arc::new(state), 1 << 20), png).await;
    assert_eq!(body["remote"]["queries"], json!([]));
    assert_eq!(body["remote"]["lookups_enabled"], false);
    let trufo = body["not_checked"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["alg"] == "ai.trufo.pawprint.watermark")
        .unwrap();
    assert!(
        trufo["reason"]
            .as_str()
            .unwrap()
            .contains("REMOTE_UPLOADS=trufo"),
        "{trufo}"
    );
}

async fn post(app: Router, png: Vec<u8>) -> Value {
    let response = app
        .oneshot(
            Request::post("/v1/detect")
                .header("content-type", "image/png")
                .body(Body::from(png))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
}
