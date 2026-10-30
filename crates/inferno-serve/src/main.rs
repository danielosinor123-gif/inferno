//! inferno-serve — the engine HTTP service (Rust side of ADR-001).
//!
//! Usage:
//!   inferno-serve --addr 127.0.0.1:8081 \
//!                --model chat=/root/models/qwen2.5-0.5b \
//!                [--blocks 256] [--budget-mb 1400]
//!
//! Routes: GET /healthz, GET /v1/models, POST /v1/models/{name}/load,
//! POST /v1/chat/completions (OpenAI-shaped; non-streaming in this slice —
//! SSE is the next pre-registered step).

mod decision;
mod http;
mod server;

use std::io::{BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Mutex;

use inferno_core::threads::ThreadPool;

use crate::http::{parse_request, ParseError, Response};
use crate::server::ServeState;

fn arg_flag(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let addr = arg_flag(&args, "--addr", "127.0.0.1:8081");
    let blocks: usize = arg_flag(&args, "--blocks", "256").parse().unwrap_or(256);
    let budget_mb: usize = arg_flag(&args, "--budget-mb", "1400")
        .parse()
        .unwrap_or(1400);

    let mut state = ServeState::new(budget_mb * 1024 * 1024, blocks);
    // --model name=path (repeatable)
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--model" {
            if let Some(spec) = args.get(i + 1) {
                if let Some((name, path)) = spec.split_once('=') {
                    state.register(name, PathBuf::from(path));
                } else {
                    eprintln!("--model expects name=path, got {spec:?}");
                    std::process::exit(2);
                }
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    if state.known_models().is_empty() {
        eprintln!("usage: inferno-serve --model name=path [--addr H:P] [--blocks N] [--budget-mb N]");
        std::process::exit(2);
    }

    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("bind {addr}: {e}");
        std::process::exit(1);
    });
    // Arc so per-connection tasks satisfy the pool's 'static captures.
    let state = std::sync::Arc::new(Mutex::new(state));
    let pool = ThreadPool::new(Some(
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
    ))
    .expect("thread pool");

    eprintln!(
        "inferno-serve on {addr} with models {:?}",
        state.lock().unwrap().known_models()
    );
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let st = std::sync::Arc::clone(&state);
        pool.spawn(move || {
            if let Err(e) = serve_connection(stream, &st) {
                // per-connection failures are logged, never fatal
                eprintln!("connection error: {e}");
            }
        });
    }
}

/// One connection: request loop until EOF, error, or `Connection: close`.
fn serve_connection(stream: TcpStream, state: &Mutex<ServeState>) -> std::io::Result<()> {
    let mut out = stream.try_clone()?;
    let mut reader = BufReader::new(stream);
    let mut buf: Vec<u8> = Vec::with_capacity(16 * 1024);
    let mut chunk = [0u8; 16 * 1024];
    loop {
        // Grow the buffer until a full request parses (or the peer stops).
        let req = loop {
            match parse_request(&buf) {
                Ok(Some(r)) => {
                    let consumed = head_end(&buf) + r.content_length().unwrap_or(0);
                    buf.drain(..consumed);
                    break r;
                }
                Ok(None) => {
                    let n = reader.read(&mut chunk)?;
                    if n == 0 {
                        return Ok(()); // clean EOF
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                Err(ParseError::Malformed(msg)) => {
                    let resp = Response::json(
                        400,
                        &serde_json::json!({"error": {"type": "invalid_request",
                            "message": format!("malformed HTTP: {msg}")}}),
                    );
                    out.write_all(&resp.to_bytes())?;
                    out.flush()?;
                    return Ok(());
                }
                Err(ParseError::Incomplete) => unreachable!("never returned"),
            }
        };

        let close = req
            .header("connection")
            .map(|v| v.eq_ignore_ascii_case("close"))
            .unwrap_or(false);
        // Streaming chat runs its own protocol (SSE, connection-close
        // framing) — the state lock is held for the generation.
        if ServeState::wants_stream(&req) {
            let mut s = state.lock().unwrap();
            if let Err(e) = s.handle_streaming(&req, &mut out) {
                eprintln!("stream error: {e}");
            }
            return Ok(());
        }
        let resp = {
            let mut s = state.lock().unwrap();
            s.handle(&req)
        };
        out.write_all(&resp.to_bytes())?;
        out.flush()?;
        if close {
            return Ok(());
        }
    }
}

/// Byte index just past the request head (after the blank line).
fn head_end(buf: &[u8]) -> usize {
    buf.windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|p| p + 4)
        .unwrap_or(buf.len())
}
