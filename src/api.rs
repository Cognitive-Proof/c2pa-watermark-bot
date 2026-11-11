//! HTTP handlers.

use std::{collections::HashMap, sync::Arc, time::Instant};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, FromRequest, Multipart, Request, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use image::DynamicImage;
use serde::Serialize;

use crate::{
    detectors::{Detection, Detector, Payload},
    manifest::{self, ManifestReport},
    registry::Registry,
};

pub struct AppState {
    pub registry: Registry,
    pub detectors: Vec<Arc<dyn Detector>>,
    /// Algorithms with a local decoder that could not be loaded, and why.
    pub unavailable: HashMap<String, String>,
}

pub fn router(state: Arc<AppState>, max_upload_bytes: usize) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/algorithms", get(algorithms))
        .route("/v1/detect", post(detect))
        .layer(DefaultBodyLimit::max(max_upload_bytes))
        .with_state(state)
}

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("no image in request body")]
    MissingImage,
    #[error("unsupported or unrecognized image format")]
    UnsupportedFormat,
    #[error("could not decode image: {0}")]
    Decode(String),
    #[error("invalid multipart body: {0}")]
    Multipart(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self {
            ApiError::MissingImage | ApiError::Multipart(_) => StatusCode::BAD_REQUEST,
            ApiError::UnsupportedFormat => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            ApiError::Decode(_) => StatusCode::UNPROCESSABLE_ENTITY,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (
            status,
            Json(serde_json::json!({ "error": self.to_string() })),
        )
            .into_response()
    }
}

#[derive(Serialize)]
struct AlgorithmStatus {
    identifier: u32,
    alg: String,
    #[serde(rename = "type")]
    kind: String,
    applies_to_images: bool,
    local_decoder: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    unavailable_reason: Option<String>,
    resolution_apis: Vec<String>,
}

async fn algorithms(State(state): State<Arc<AppState>>) -> Json<Vec<AlgorithmStatus>> {
    let loaded: Vec<&str> = state.detectors.iter().map(|d| d.alg()).collect();
    Json(
        state
            .registry
            .all()
            .iter()
            .map(|a| AlgorithmStatus {
                identifier: a.identifier,
                alg: a.alg.clone(),
                kind: a.kind.clone(),
                applies_to_images: a.applies_to_images(),
                local_decoder: loaded.contains(&a.alg.as_str()),
                unavailable_reason: state.unavailable.get(&a.alg).cloned(),
                resolution_apis: a.soft_binding_resolution_apis.clone(),
            })
            .collect(),
    )
}

#[derive(Serialize)]
pub struct DetectResponse {
    pub media_type: String,
    pub width: u32,
    pub height: u32,
    /// The `alg` of every watermark found by a local decoder.
    pub found: Vec<String>,
    pub watermarks: Vec<WatermarkResult>,
    pub c2pa: C2paResult,
    /// Image algorithms in the list that were not checked, and why.
    pub not_checked: Vec<NotChecked>,
}

#[derive(Serialize)]
pub struct WatermarkResult {
    pub alg: String,
    pub found: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<Payload>,
    /// The detector's confidence score, where it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    /// Whether the payload matches a soft-binding assertion for the same
    /// algorithm in the image's manifest. Absent when there is no such assertion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matches_manifest: Option<bool>,
    /// Caveats about how the detector relates to the registered algorithm.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub elapsed_ms: u128,
}

#[derive(Serialize)]
pub struct C2paResult {
    #[serde(flatten)]
    pub report: ManifestReport,
    /// Registry details for each `alg` named in a soft-binding assertion.
    pub soft_binding_algorithms: Vec<SoftBindingAlgorithm>,
}

#[derive(Serialize)]
pub struct SoftBindingAlgorithm {
    pub alg: String,
    pub registered: bool,
    pub local_decoder: bool,
    pub resolution_apis: Vec<String>,
}

#[derive(Serialize)]
pub struct NotChecked {
    pub alg: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub reason: String,
    pub resolution_apis: Vec<String>,
}

async fn read_image_bytes(req: Request) -> Result<Bytes, ApiError> {
    let is_multipart = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("multipart/form-data"));

    let bytes = if is_multipart {
        let mut multipart = Multipart::from_request(req, &())
            .await
            .map_err(|e| ApiError::Multipart(e.body_text()))?;
        let mut image = None;
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|e| ApiError::Multipart(e.body_text()))?
        {
            if field.name() == Some("image") {
                image = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| ApiError::Multipart(e.body_text()))?,
                );
                break;
            }
        }
        image.ok_or(ApiError::MissingImage)?
    } else {
        Bytes::from_request(req, &())
            .await
            .map_err(|e| ApiError::Multipart(e.body_text()))?
    };

    if bytes.is_empty() {
        return Err(ApiError::MissingImage);
    }
    Ok(bytes)
}

async fn detect(
    State(state): State<Arc<AppState>>,
    req: Request,
) -> Result<Json<DetectResponse>, ApiError> {
    let bytes = read_image_bytes(req).await?;
    let format = image::guess_format(&bytes).map_err(|_| ApiError::UnsupportedFormat)?;
    let media_type = format.to_mime_type().to_owned();

    let img = {
        let bytes = bytes.clone();
        tokio::task::spawn_blocking(move || image::load_from_memory_with_format(&bytes, format))
            .await
            .map_err(|e| ApiError::Internal(e.to_string()))?
            .map_err(|e| ApiError::Decode(e.to_string()))?
    };
    let img = Arc::new(img);

    let manifest_task = {
        let media_type = media_type.clone();
        tokio::task::spawn_blocking(move || manifest::read(&media_type, &bytes))
    };
    let detector_tasks: Vec<_> = state
        .detectors
        .iter()
        .map(|detector| {
            let detector = Arc::clone(detector);
            let img = Arc::clone(&img);
            tokio::task::spawn_blocking(move || run_detector(detector.as_ref(), &img))
        })
        .collect();

    let report = manifest_task
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    let mut watermarks = Vec::with_capacity(detector_tasks.len());
    for (task, detector) in detector_tasks.into_iter().zip(&state.detectors) {
        let mut result = task.await.map_err(|e| ApiError::Internal(e.to_string()))?;
        result.matches_manifest = match_manifest(&report, detector.as_ref(), &result);
        watermarks.push(result);
    }

    Ok(Json(build_response(
        &state, media_type, &img, watermarks, report,
    )))
}

fn run_detector(detector: &dyn Detector, img: &DynamicImage) -> WatermarkResult {
    let start = Instant::now();
    let outcome = detector.detect(img);
    let elapsed_ms = start.elapsed().as_millis();
    let (detection, error) = match outcome {
        Ok(detection) => (detection, None),
        Err(err) => (Detection::default(), Some(err.to_string())),
    };
    WatermarkResult {
        alg: detector.alg().to_owned(),
        found: detection.payload.is_some(),
        payload: detection.payload,
        score: detection.score,
        matches_manifest: None,
        note: detector.note().map(str::to_owned),
        error,
        elapsed_ms,
    }
}

fn match_manifest(
    report: &ManifestReport,
    detector: &dyn Detector,
    result: &WatermarkResult,
) -> Option<bool> {
    let values: Vec<&str> = report
        .soft_bindings
        .iter()
        .filter(|b| b.alg.as_deref() == Some(result.alg.as_str()))
        .flat_map(|b| b.values.iter().map(String::as_str))
        .collect();
    if values.is_empty() {
        return None;
    }
    Some(
        result
            .payload
            .as_ref()
            .is_some_and(|payload| values.iter().any(|v| detector.matches_value(v, payload))),
    )
}

fn build_response(
    state: &AppState,
    media_type: String,
    img: &DynamicImage,
    watermarks: Vec<WatermarkResult>,
    report: ManifestReport,
) -> DetectResponse {
    let loaded: Vec<&str> = state.detectors.iter().map(|d| d.alg()).collect();

    let mut soft_binding_algs: Vec<String> = report
        .soft_bindings
        .iter()
        .filter_map(|b| b.alg.clone())
        .collect();
    soft_binding_algs.sort();
    soft_binding_algs.dedup();
    let soft_binding_algorithms = soft_binding_algs
        .into_iter()
        .map(|alg| {
            let entry = state.registry.get(&alg);
            SoftBindingAlgorithm {
                registered: entry.is_some(),
                local_decoder: loaded.contains(&alg.as_str()),
                resolution_apis: entry
                    .map(|e| e.soft_binding_resolution_apis.clone())
                    .unwrap_or_default(),
                alg,
            }
        })
        .collect();

    let not_checked = state
        .registry
        .all()
        .iter()
        .filter(|a| a.applies_to_images() && !loaded.contains(&a.alg.as_str()))
        .map(|a| NotChecked {
            alg: a.alg.clone(),
            kind: a.kind.clone(),
            reason: state
                .unavailable
                .get(&a.alg)
                .cloned()
                .unwrap_or_else(|| "no public decoder".to_owned()),
            resolution_apis: a.soft_binding_resolution_apis.clone(),
        })
        .collect();

    DetectResponse {
        media_type,
        width: img.width(),
        height: img.height(),
        found: watermarks
            .iter()
            .filter(|w| w.found)
            .map(|w| w.alg.clone())
            .collect(),
        watermarks,
        c2pa: C2paResult {
            report,
            soft_binding_algorithms,
        },
        not_checked,
    }
}
