//! Fail-closed transport and run-identity configuration.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::net::IpAddr;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rand::RngCore;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::server::WebPkiClientVerifier;
use rustls::{RootCertStore, ServerConfig};
use sha2::{Digest, Sha256};
use tokio_rustls::TlsAcceptor;

pub const RUN_ID_BYTES: usize = 32;

#[derive(Clone)]
pub struct TlsServer {
    pub acceptor: TlsAcceptor,
    fingerprints: Arc<HashMap<u32, [u8; 32]>>,
}

impl TlsServer {
    pub fn expected_fingerprint(&self, learner_id: u32) -> Option<[u8; 32]> {
        self.fingerprints.get(&learner_id).copied()
    }
}

#[derive(Clone)]
pub enum TransportSecurity {
    PlaintextLoopback,
    Tls(TlsServer),
}

fn read_certs(path: &Path, label: &str) -> Result<Vec<CertificateDer<'static>>> {
    let file = File::open(path).with_context(|| format!("open {label} certificate file"))?;
    let mut reader = BufReader::new(file);
    let certs = rustls_pemfile::certs(&mut reader)
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("parse {label} certificate file"))?;
    if certs.is_empty() {
        bail!("{label} certificate file contains no certificates");
    }
    Ok(certs)
}

fn read_private_key(path: &Path) -> Result<PrivateKeyDer<'static>> {
    if std::fs::symlink_metadata(path)?.file_type().is_symlink() {
        bail!("TLS server private key must not be a symbolic link");
    }
    let mode = std::fs::metadata(path)?.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        bail!("TLS server private key must not be accessible by group or other users");
    }
    let file = File::open(path).context("open TLS server private-key file")?;
    let mut reader = BufReader::new(file);
    rustls_pemfile::private_key(&mut reader)
        .context("parse TLS server private-key file")?
        .context("TLS server private-key file contains no private key")
}

fn decode_fingerprint(value: &str) -> Result<[u8; 32]> {
    let normalized: String = value
        .chars()
        .filter(|character| *character != ':')
        .collect::<String>()
        .to_ascii_lowercase();
    if normalized.len() != 64 || !normalized.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("client certificate fingerprint must be exactly 32 SHA-256 bytes");
    }
    let mut output = [0u8; 32];
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&normalized[index * 2..index * 2 + 2], 16)
            .context("decode client certificate fingerprint")?;
    }
    Ok(output)
}

fn load_fingerprint_allowlist(path: &Path, learners: u32) -> Result<HashMap<u32, [u8; 32]>> {
    let content = std::fs::read_to_string(path).context("read TLS client fingerprint allowlist")?;
    let mut allowlist = HashMap::new();
    let mut unique = HashSet::new();
    for (line_index, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 2 {
            bail!(
                "TLS client fingerprint allowlist line {} must contain learner_id and SHA-256",
                line_index + 1
            );
        }
        let learner_id: u32 = fields[0]
            .parse()
            .with_context(|| format!("invalid learner id on allowlist line {}", line_index + 1))?;
        if learner_id >= learners {
            bail!("allowlisted learner id {learner_id} is outside configured range 0..{learners}");
        }
        let fingerprint = decode_fingerprint(fields[1])?;
        if allowlist.insert(learner_id, fingerprint).is_some() {
            bail!("duplicate learner id {learner_id} in TLS client fingerprint allowlist");
        }
        if !unique.insert(fingerprint) {
            bail!("every learner must have a distinct TLS client certificate fingerprint");
        }
    }
    if allowlist.len() != learners as usize {
        let missing: Vec<_> = (0..learners)
            .filter(|learner_id| !allowlist.contains_key(learner_id))
            .collect();
        bail!("TLS client fingerprint allowlist is incomplete; missing learner ids {missing:?}");
    }
    Ok(allowlist)
}

pub fn configure_transport(
    bind_address: IpAddr,
    allow_insecure_loopback: bool,
    server_cert: Option<PathBuf>,
    server_key: Option<PathBuf>,
    client_ca: Option<PathBuf>,
    client_fingerprints: Option<PathBuf>,
    learners: u32,
) -> Result<TransportSecurity> {
    let configured = [
        server_cert.is_some(),
        server_key.is_some(),
        client_ca.is_some(),
        client_fingerprints.is_some(),
    ];
    let tls_configured = configured.iter().all(|value| *value);
    if configured.iter().any(|value| *value) && !tls_configured {
        bail!(
            "TLS requires all of --tls-cert, --tls-key, --tls-client-ca, and --tls-client-fingerprints"
        );
    }
    if !tls_configured {
        if !bind_address.is_loopback() {
            bail!("non-loopback listeners require TLS 1.3 mutual authentication");
        }
        if !allow_insecure_loopback {
            bail!(
                "plaintext is disabled; use complete TLS configuration or the explicit --allow-insecure-loopback development profile"
            );
        }
        return Ok(TransportSecurity::PlaintextLoopback);
    }
    if allow_insecure_loopback {
        bail!("--allow-insecure-loopback cannot be combined with TLS configuration");
    }

    let certs = read_certs(server_cert.as_deref().unwrap(), "TLS server")?;
    let key = read_private_key(server_key.as_deref().unwrap())?;
    let mut roots = RootCertStore::empty();
    for certificate in read_certs(client_ca.as_deref().unwrap(), "TLS client CA")? {
        roots
            .add(certificate)
            .context("add TLS client CA certificate to trust store")?;
    }
    let verifier = WebPkiClientVerifier::builder(Arc::new(roots))
        .build()
        .context("build mutual-TLS client certificate verifier")?;
    let config = ServerConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
        .with_client_cert_verifier(verifier)
        .with_single_cert(certs, key)
        .context("configure TLS server certificate and private key")?;
    let fingerprints =
        load_fingerprint_allowlist(client_fingerprints.as_deref().unwrap(), learners)?;
    Ok(TransportSecurity::Tls(TlsServer {
        acceptor: TlsAcceptor::from(Arc::new(config)),
        fingerprints: Arc::new(fingerprints),
    }))
}

pub fn peer_fingerprint(certificate: &CertificateDer<'_>) -> [u8; 32] {
    Sha256::digest(certificate.as_ref()).into()
}

fn validate_run_id_file_mode(path: &Path) -> Result<()> {
    if std::fs::symlink_metadata(path)?.file_type().is_symlink() {
        bail!("run ID file must not be a symbolic link");
    }
    let mode = std::fs::metadata(path)?.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        bail!("run ID file must not be accessible by group or other users");
    }
    Ok(())
}

fn read_run_id(path: &Path) -> Result<[u8; RUN_ID_BYTES]> {
    validate_run_id_file_mode(path)?;
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("run ID file must contain exactly 32 raw bytes"))
}

pub fn load_or_create_run_id(path: &Path, resume: bool) -> Result<[u8; RUN_ID_BYTES]> {
    if path.exists() {
        return read_run_id(path).context("read durable run ID");
    }
    if resume {
        bail!("--resume requires the durable run ID file to already exist");
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut run_id = [0u8; RUN_ID_BYTES];
    rand::rngs::OsRng.fill_bytes(&mut run_id);
    let mut file = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            return read_run_id(path).context("read concurrently created durable run ID");
        }
        Err(error) => return Err(error).context("create durable run ID file"),
    };
    file.write_all(&run_id)?;
    file.sync_all()?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(run_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_requires_distinct_complete_identities() {
        let directory = std::env::temp_dir().join(format!(
            "yeto-security-allowlist-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("allowlist");
        std::fs::write(
            &path,
            format!("0 {}\n1 {}\n", "01".repeat(32), "02".repeat(32)),
        )
        .unwrap();
        assert_eq!(load_fingerprint_allowlist(&path, 2).unwrap().len(), 2);
        std::fs::write(
            &path,
            format!("0 {}\n1 {}\n", "01".repeat(32), "01".repeat(32)),
        )
        .unwrap();
        assert!(load_fingerprint_allowlist(&path, 2)
            .unwrap_err()
            .to_string()
            .contains("distinct"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn durable_run_id_is_random_reused_and_private() {
        let directory = std::env::temp_dir().join(format!(
            "yeto-security-run-id-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let path = directory.join("run-id");
        let first = load_or_create_run_id(&path, false).unwrap();
        assert_ne!(first, [0u8; RUN_ID_BYTES]);
        assert_eq!(load_or_create_run_id(&path, true).unwrap(), first);
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn durable_run_id_rejects_missing_resume_bad_length_and_unsafe_mode() {
        let directory = std::env::temp_dir().join(format!(
            "yeto-security-invalid-run-id-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let path = directory.join("run-id");
        let error = load_or_create_run_id(&path, true).unwrap_err();
        assert!(error.to_string().contains("--resume requires"));

        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(&path, [0u8; RUN_ID_BYTES - 1]).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let error = load_or_create_run_id(&path, false).unwrap_err();
        assert!(format!("{error:#}").contains("exactly 32 raw bytes"));

        std::fs::write(&path, [0u8; RUN_ID_BYTES]).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        let error = load_or_create_run_id(&path, false).unwrap_err();
        assert!(format!("{error:#}").contains("group or other users"));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
