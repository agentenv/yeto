//! Line-oriented exact-f32 transcendental oracle for the independent CPLG replay.
//!
//! Production CPLG uses the pinned `libm = 0.2.15` crate.  Python's geometric
//! reference remains independent of the production state machine, but it must
//! use these exact transcendental results before its candidate bytes can be a
//! bit-parity fixture.  This helper exposes only `atan2f`, `sinf`, and `cosf`
//! over raw IEEE-754 bits.  It owns no optimizer state and performs no vector
//! arithmetic.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{self, BufRead, BufWriter, Write};

const SCHEMA: &str = "cplg_rust_libm_oracle_v1";
const LIBM_VERSION: &str = "0.2.15";
const ANGLE_CAP_BITS: u32 = 0x3e7a_dbb0;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: u64,
    op: String,
    x_bits: String,
    #[serde(default)]
    y_bits: Option<String>,
}

fn parse_finite_f32_bits(text: &str, field: &str) -> Result<f32> {
    if text.len() != 8
        || !text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{field} must be exactly eight lowercase hexadecimal digits");
    }
    let bits = u32::from_str_radix(text, 16).context("invalid f32 bits")?;
    let value = f32::from_bits(bits);
    if !value.is_finite() {
        bail!("{field} must encode a finite f32");
    }
    Ok(value)
}

fn response(request: Request) -> Result<Value> {
    let x = parse_finite_f32_bits(&request.x_bits, "x_bits")?;
    let result = match request.op.as_str() {
        "sinf" => {
            if request.y_bits.is_some() {
                bail!("sinf forbids y_bits");
            }
            libm::sinf(x)
        }
        "cosf" => {
            if request.y_bits.is_some() {
                bail!("cosf forbids y_bits");
            }
            libm::cosf(x)
        }
        "atan2f" => {
            let y = parse_finite_f32_bits(
                request
                    .y_bits
                    .as_deref()
                    .context("atan2f requires y_bits")?,
                "y_bits",
            )?;
            libm::atan2f(y, x)
        }
        other => bail!("unsupported operation {other:?}"),
    };
    if !result.is_finite() {
        bail!("oracle operation produced nonfinite f32");
    }
    Ok(json!({
        "schema": SCHEMA,
        "id": request.id,
        "result_bits": format!("{:08x}", result.to_bits()),
    }))
}

fn require_object_without_duplicate_keys(line: &str) -> Result<Request> {
    // serde_json's normal map representation cannot retain duplicate keys.
    // Deserialize once to a BTreeMap to enforce the closed key set before the
    // strongly typed, deny-unknown-fields decode.
    let fields: BTreeMap<String, Value> =
        serde_json::from_str(line).context("request must be one JSON object")?;
    let expected = match fields.get("op").and_then(Value::as_str) {
        Some("atan2f") => ["id", "op", "x_bits", "y_bits"].as_slice(),
        _ => ["id", "op", "x_bits"].as_slice(),
    };
    let actual: Vec<&str> = fields.keys().map(String::as_str).collect();
    let mut expected_sorted = expected.to_vec();
    expected_sorted.sort_unstable();
    if actual != expected_sorted {
        bail!("request fields differ from the closed protocol");
    }
    serde_json::from_str(line).context("invalid oracle request")
}

fn write_json_line(writer: &mut BufWriter<io::StdoutLock<'_>>, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value).context("write JSON response")?;
    writer.write_all(b"\n").context("write response newline")?;
    writer.flush().context("flush response")?;
    Ok(())
}

fn main() -> Result<()> {
    if libm::atan2f(0.25, 1.0).to_bits() != ANGLE_CAP_BITS
        || libm::sinf(f32::from_bits(ANGLE_CAP_BITS)).to_bits() != 0x3e78_5b42
        || libm::cosf(f32::from_bits(ANGLE_CAP_BITS)).to_bits() != 0x3f78_5b42
    {
        bail!("pinned libm self-test failed");
    }

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    write_json_line(
        &mut writer,
        &json!({
            "schema": SCHEMA,
            "status": "ready",
            "libm_version": LIBM_VERSION,
            "angle_cap_bits": format!("{ANGLE_CAP_BITS:08x}"),
            "cap_sinf_bits": "3e785b42",
            "cap_cosf_bits": "3f785b42",
        }),
    )?;

    for (line_number, line) in stdin.lock().lines().enumerate() {
        let line = line.with_context(|| format!("read request line {}", line_number + 1))?;
        if line.is_empty() {
            bail!("empty request line {}", line_number + 1);
        }
        let request = require_object_without_duplicate_keys(&line)
            .with_context(|| format!("request line {}", line_number + 1))?;
        let result =
            response(request).with_context(|| format!("request line {}", line_number + 1))?;
        write_json_line(&mut writer, &result)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_cap_values_match_production_contract() {
        let cap = f32::from_bits(ANGLE_CAP_BITS);
        assert_eq!(libm::atan2f(0.25, 1.0).to_bits(), ANGLE_CAP_BITS);
        assert_eq!(libm::sinf(cap).to_bits(), 0x3e78_5b42);
        assert_eq!(libm::cosf(cap).to_bits(), 0x3f78_5b42);
    }

    #[test]
    fn closed_request_contract_is_strict() {
        assert!(require_object_without_duplicate_keys(
            r#"{"id":1,"op":"sinf","x_bits":"3e7adbb0"}"#
        )
        .is_ok());
        assert!(require_object_without_duplicate_keys(
            r#"{"id":1,"op":"sinf","x_bits":"3e7adbb0","extra":0}"#
        )
        .is_err());
        assert!(require_object_without_duplicate_keys(
            r#"{"id":1,"op":"atan2f","x_bits":"3f800000"}"#
        )
        .is_err());
    }
}
