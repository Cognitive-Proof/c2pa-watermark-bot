pub mod api;
pub mod detectors;
pub mod manifest;
pub mod registry;
pub mod remote;

use std::{collections::HashMap, path::Path, sync::Arc};

use detectors::{
    Detector,
    invismark::{self, InvisMarkDetector},
    trustmark::{self as tm, TrustmarkDetector},
    videoseal::{self as vs, VideoSealDetector},
};
use registry::Registry;
use remote::RemoteConfig;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// Loads the algorithm list, every local detector whose model files are
/// present, and the resolution API client.
pub fn load_state(models_dir: &Path, remote: RemoteConfig) -> Result<api::AppState, Error> {
    let registry = Registry::load()?;
    let mut detectors: Vec<Arc<dyn Detector>> = Vec::new();
    let mut unavailable = HashMap::new();

    for variant in tm::VARIANTS {
        let alg = TrustmarkDetector::alg_for(variant);
        if !TrustmarkDetector::models_present(models_dir, variant) {
            unavailable.insert(
                alg,
                format!("model files not found in {}", models_dir.display()),
            );
            continue;
        }
        match TrustmarkDetector::load(models_dir, variant) {
            Ok(detector) => {
                tracing::info!(%alg, "loaded detector");
                detectors.push(Arc::new(detector));
            }
            Err(err) => {
                tracing::warn!(%alg, %err, "failed to load detector");
                unavailable.insert(alg, format!("failed to load model: {err}"));
            }
        }
    }

    for model in vs::Model::ALL {
        let alg = model.alg().to_owned();
        if !models_dir.join(model.filename()).is_file() {
            unavailable.insert(
                alg,
                format!(
                    "{} not found in {} (see scripts/export_videoseal_onnx.py)",
                    model.filename(),
                    models_dir.display()
                ),
            );
            continue;
        }
        match VideoSealDetector::load(models_dir, model) {
            Ok(detector) => {
                tracing::info!(%alg, "loaded detector");
                detectors.push(Arc::new(detector));
            }
            Err(err) => {
                tracing::warn!(%alg, %err, "failed to load detector");
                unavailable.insert(alg, format!("failed to load model: {err}"));
            }
        }
    }

    let alg = invismark::ALG.to_owned();
    if !models_dir.join(invismark::MODEL_FILENAME).is_file() {
        unavailable.insert(
            alg,
            format!(
                "{} not found in {} (see scripts/export_invismark_onnx.py)",
                invismark::MODEL_FILENAME,
                models_dir.display()
            ),
        );
    } else {
        match InvisMarkDetector::load(models_dir) {
            Ok(detector) => {
                tracing::info!(%alg, "loaded detector");
                detectors.push(Arc::new(detector));
            }
            Err(err) => {
                tracing::warn!(%alg, %err, "failed to load detector");
                unavailable.insert(alg, format!("failed to load model: {err}"));
            }
        }
    }

    for alg in unavailable.keys() {
        tracing::warn!(%alg, reason = %unavailable[alg], "detector unavailable");
    }

    let remote = remote::Client::new(&registry, remote)?;
    tracing::info!(
        lookups = remote.config.lookups,
        uploads = ?remote.upload_services().map(|s| &s.name).collect::<Vec<_>>(),
        "remote resolution APIs"
    );

    Ok(api::AppState {
        registry,
        detectors,
        unavailable,
        remote,
    })
}
