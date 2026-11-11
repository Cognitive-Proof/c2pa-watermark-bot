use std::{net::SocketAddr, path::PathBuf, sync::Arc};

use c2pa_watermark_bot::{api, load_state};
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,ort=warn")),
        )
        .init();

    let addr: SocketAddr = std::env::var("BIND_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8080".to_owned())
        .parse()?;
    let models_dir =
        PathBuf::from(std::env::var("MODELS_DIR").unwrap_or_else(|_| "models".to_owned()));
    let max_upload_mb: usize = std::env::var("MAX_UPLOAD_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25);

    let state = tokio::task::spawn_blocking(move || load_state(&models_dir)).await??;
    let app =
        api::router(Arc::new(state), max_upload_mb * 1024 * 1024).layer(TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
