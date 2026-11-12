//! Queries vendors' C2PA Soft Binding Resolution APIs
//! (<https://spec.c2pa.org/specifications/specifications/2.2/softbinding/Decoupled.html>).
//!
//! Both kinds of query send data to third parties, so both are off unless
//! configured:
//!
//! - **Lookups** (`GET /matches/byBinding`) send a value derived from the
//!   image, never the image: a payload a local detector decoded, or the file's
//!   SHA-256 for Monolith, whose fingerprint is that hash.
//! - **Uploads** (`POST /matches/byContent`) send the image itself, to the
//!   services named in the configuration.

use std::{collections::BTreeMap, time::Duration, time::Instant};

use base64::prelude::{BASE64_STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::registry::Registry;

/// Resolution APIs that aren't in the algorithm list. Reconize publishes this
/// one at <https://reconize.me/softbinding.html>.
const EXTRA_APIS: [(&str, &str); 1] = [(
    "me.reconize.videoseal.1",
    "https://relay.reconize.me/softbinding/v1",
)];

/// Services that resolve uploaded images. Checked in 2026-11: BlockFact's
/// upload endpoint is a stub that asks for an extracted value, Reconize
/// refuses uploads, and Monolith is looked up by hash instead.
pub const UPLOAD_SERVICES: [&str; 3] = ["trufo", "aiwatermark", "etchv"];

/// Fingerprints that are a plain hash of the file, so they can be computed
/// locally and looked up without uploading.
const SHA256_FINGERPRINTS: [&str; 1] = ["com.joinmonolith.sha256"];

const TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Debug, Clone, Default)]
pub struct RemoteConfig {
    /// Send decoded payloads and file hashes to resolution APIs.
    pub lookups: bool,
    /// Services to upload images to, by name (see [`UPLOAD_SERVICES`]).
    pub uploads: Vec<String>,
}

impl RemoteConfig {
    /// Reads `REMOTE_LOOKUPS` (`1`/`true`) and `REMOTE_UPLOADS` (a comma-separated
    /// list of service names, or `all`).
    pub fn from_env() -> Self {
        let lookups = std::env::var("REMOTE_LOOKUPS")
            .is_ok_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"));
        let uploads = std::env::var("REMOTE_UPLOADS").unwrap_or_default();
        let uploads = if uploads.trim().eq_ignore_ascii_case("all") {
            UPLOAD_SERVICES.iter().map(|s| s.to_string()).collect()
        } else {
            uploads
                .split(',')
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect()
        };
        Self { lookups, uploads }
    }
}

/// One resolution API and the image algorithms it resolves.
#[derive(Debug, Clone)]
pub struct Service {
    /// A short name from the host, such as `trufo` for `c2pa.trufo.ai`.
    pub name: String,
    pub base_url: String,
    pub algs: Vec<String>,
}

impl Service {
    pub fn accepts_uploads(&self) -> bool {
        UPLOAD_SERVICES.contains(&self.name.as_str())
    }
}

/// The resolution APIs for image algorithms in the list, plus [`EXTRA_APIS`].
pub fn services(registry: &Registry) -> Vec<Service> {
    let mut by_url: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for alg in registry.all().iter().filter(|a| a.applies_to_images()) {
        for url in &alg.soft_binding_resolution_apis {
            by_url.entry(url.clone()).or_default().push(alg.alg.clone());
        }
    }
    for (alg, url) in EXTRA_APIS {
        by_url
            .entry(url.to_owned())
            .or_default()
            .push(alg.to_owned());
    }
    by_url
        .into_iter()
        .map(|(base_url, algs)| Service {
            name: service_name(&base_url),
            base_url: base_url.trim_end_matches('/').to_owned(),
            algs,
        })
        .collect()
}

fn service_name(url: &str) -> String {
    let host = url
        .split("://")
        .nth(1)
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or_default();
    let labels: Vec<&str> = host.split('.').collect();
    labels
        .len()
        .checked_sub(2)
        .map(|i| labels[i])
        .unwrap_or(host)
        .to_owned()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Match {
    pub manifest_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub similarity_score: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct QueryResult {
    #[serde(default)]
    matches: Vec<Match>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RemoteQuery {
    pub service: String,
    pub base_url: String,
    /// The algorithm queried; absent for an upload that covers all of the
    /// service's algorithms.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alg: Option<String>,
    /// `byBinding` (a lookup) or `byContent` (an upload).
    pub method: &'static str,
    /// The base64 value sent, for lookups.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    pub matches: Vec<Match>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub elapsed_ms: u128,
}

impl RemoteQuery {
    /// The algorithms this query checked.
    pub fn algs<'a>(&'a self, services: &'a [Service]) -> Vec<&'a str> {
        match &self.alg {
            Some(alg) => vec![alg.as_str()],
            None => services
                .iter()
                .find(|s| s.base_url == self.base_url)
                .map(|s| s.algs.iter().map(String::as_str).collect())
                .unwrap_or_default(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RemoteReport {
    pub lookups_enabled: bool,
    pub uploads_enabled: Vec<String>,
    pub queries: Vec<RemoteQuery>,
}

/// A decoded payload to look up.
pub struct Lookup {
    pub alg: String,
    pub bytes: Vec<u8>,
}

pub struct Client {
    http: reqwest::Client,
    pub services: Vec<Service>,
    pub config: RemoteConfig,
}

impl Client {
    pub fn new(registry: &Registry, config: RemoteConfig) -> reqwest::Result<Self> {
        Self::with_services(services(registry), config)
    }

    /// A client for the given services instead of those in the algorithm list.
    pub fn with_services(services: Vec<Service>, config: RemoteConfig) -> reqwest::Result<Self> {
        for name in &config.uploads {
            match services.iter().find(|s| &s.name == name) {
                Some(s) if s.accepts_uploads() => {}
                Some(_) => tracing::warn!(%name, "service does not resolve uploads; ignoring"),
                None => tracing::warn!(%name, "unknown upload service; ignoring"),
            }
        }
        let http = reqwest::Client::builder()
            .timeout(TIMEOUT)
            .user_agent(concat!("c2pa-watermark-bot/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self {
            http,
            services,
            config,
        })
    }

    fn service_for(&self, alg: &str) -> Option<&Service> {
        self.services
            .iter()
            .find(|s| s.algs.iter().any(|a| a == alg))
    }

    pub fn upload_services(&self) -> impl Iterator<Item = &Service> {
        self.services
            .iter()
            .filter(|s| s.accepts_uploads() && self.config.uploads.contains(&s.name))
    }

    /// Runs the configured lookups and uploads for one image.
    pub async fn query(
        &self,
        media_type: &str,
        file: &bytes::Bytes,
        decoded: Vec<Lookup>,
    ) -> RemoteReport {
        let mut tasks = Vec::new();

        if self.config.lookups {
            let hashes = SHA256_FINGERPRINTS.iter().map(|alg| Lookup {
                alg: alg.to_string(),
                bytes: Sha256::digest(file).to_vec(),
            });
            for lookup in decoded.into_iter().chain(hashes) {
                if let Some(service) = self.service_for(&lookup.alg) {
                    tasks.push(tokio::spawn(by_binding(
                        self.http.clone(),
                        service.clone(),
                        lookup.alg,
                        BASE64_STANDARD.encode(&lookup.bytes),
                    )));
                }
            }
        }

        for service in self.upload_services() {
            // Name the algorithm when there is only one; otherwise let the
            // service check all of its own.
            let alg = match service.algs.as_slice() {
                [alg] => Some(alg.clone()),
                _ => None,
            };
            tasks.push(tokio::spawn(by_content(
                self.http.clone(),
                service.clone(),
                alg,
                media_type.to_owned(),
                file.clone(),
            )));
        }

        let mut queries = Vec::with_capacity(tasks.len());
        for task in tasks {
            match task.await {
                Ok(query) => queries.push(query),
                Err(err) => tracing::warn!(%err, "remote query task failed"),
            }
        }
        RemoteReport {
            lookups_enabled: self.config.lookups,
            uploads_enabled: self.upload_services().map(|s| s.name.clone()).collect(),
            queries,
        }
    }
}

async fn by_binding(
    http: reqwest::Client,
    service: Service,
    alg: String,
    value: String,
) -> RemoteQuery {
    let start = Instant::now();
    let request = http
        .get(format!("{}/matches/byBinding", service.base_url))
        .query(&[("alg", alg.as_str()), ("value", value.as_str())]);
    let (matches, error) = send(request).await;
    RemoteQuery {
        service: service.name,
        base_url: service.base_url,
        alg: Some(alg),
        method: "byBinding",
        value: Some(value),
        matches,
        error,
        elapsed_ms: start.elapsed().as_millis(),
    }
}

async fn by_content(
    http: reqwest::Client,
    service: Service,
    alg: Option<String>,
    media_type: String,
    file: bytes::Bytes,
) -> RemoteQuery {
    let start = Instant::now();
    let mut request = http
        .post(format!("{}/matches/byContent", service.base_url))
        .header(reqwest::header::CONTENT_TYPE, media_type)
        .body(file);
    if let Some(alg) = &alg {
        request = request.query(&[("alg", alg.as_str())]);
    }
    let (matches, error) = send(request).await;
    RemoteQuery {
        service: service.name,
        base_url: service.base_url,
        alg,
        method: "byContent",
        value: None,
        matches,
        error,
        elapsed_ms: start.elapsed().as_millis(),
    }
}

async fn send(request: reqwest::RequestBuilder) -> (Vec<Match>, Option<String>) {
    let response = match request.send().await {
        Ok(response) => response,
        Err(err) => return (Vec::new(), Some(err.to_string())),
    };
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        let body: String = body.chars().take(300).collect();
        return (Vec::new(), Some(format!("HTTP {status}: {body}")));
    }
    match serde_json::from_str::<QueryResult>(&body) {
        Ok(result) => (result.matches, None),
        Err(err) => (Vec::new(), Some(format!("invalid response: {err}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_services_by_host() {
        assert_eq!(service_name("https://c2pa.trufo.ai/v1"), "trufo");
        assert_eq!(
            service_name("https://aiwatermark.com/api/v1"),
            "aiwatermark"
        );
        assert_eq!(
            service_name("https://relay.reconize.me/softbinding/v1"),
            "reconize"
        );
    }

    #[test]
    fn groups_image_algorithms_by_service() {
        let registry = Registry::load().unwrap();
        let services = services(&registry);
        let trufo = services.iter().find(|s| s.name == "trufo").unwrap();
        assert_eq!(
            trufo.algs,
            [
                "ai.trufo.pawprint.watermark",
                "ai.trufo.pawprint.fingerprint"
            ]
        );
        let aiwatermark = services.iter().find(|s| s.name == "aiwatermark").unwrap();
        // The audio and video algorithms share the URL but aren't image algorithms.
        assert_eq!(aiwatermark.algs, ["com.aiwatermark.pixelseal.1"]);
        assert!(services.iter().any(|s| s.name == "reconize"));
        for name in UPLOAD_SERVICES {
            assert!(services.iter().any(|s| s.name == name), "{name}");
        }
    }

    #[test]
    fn parses_matches() {
        let result: QueryResult = serde_json::from_str(
            r#"{"matches":[{"manifestId":"urn:x","similarityScore":98},{"manifestId":"y","endpoint":"https://e"}]}"#,
        )
        .unwrap();
        assert_eq!(result.matches.len(), 2);
        assert_eq!(result.matches[0].similarity_score, Some(98.0));
        assert_eq!(result.matches[1].endpoint.as_deref(), Some("https://e"));
    }
}
