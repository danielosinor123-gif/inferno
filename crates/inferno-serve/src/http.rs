//! Minimal HTTP/1.1 message layer — std only, no frameworks (the house
//! constraint, ADR-001). Scope: request line + headers + Content-Length
//! body; responses always Content-Length-delimited. Chunked bodies,
//! compression, and keep-alive subtleties beyond close/keep-alive are
//! deliberately absent — the engine service boundary doesn't need them.

/// A parsed HTTP request.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub method: String,
    /// Path without the query string.
    pub path: String,
    /// Query parameters (percent-decoding not needed for our routes).
    pub query: Vec<(String, String)>,
    /// Header names lowercased.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

#[derive(Debug, PartialEq)]
pub enum ParseError {
    /// Buffer holds an incomplete request — read more.
    Incomplete,
    Malformed(String),
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        let n = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| *k == n)
        .map(|(_, v)| v.as_str())
    }

    pub fn content_length(&self) -> Option<usize> {
        self.header("content-length").and_then(|v| v.parse().ok())
    }
}

/// Try to parse one request from the front of `buf`. `Ok(None)` means the
/// buffer is an incomplete request (caller reads more); `Err` means the
/// bytes can never become a valid request (caller answers 400).
pub fn parse_request(buf: &[u8]) -> Result<Option<Request>, ParseError> {
    let Some(end) = find_subslice(buf, b"\r\n\r\n") else {
        // A head longer than this is never one of ours.
        if buf.len() > 64 * 1024 {
            return Err(ParseError::Malformed("header section too large".into()));
        }
        return Ok(None);
    };
    let head = &buf[..end];
    let head_str = std::str::from_utf8(head)
        .map_err(|_| ParseError::Malformed("request line is not UTF-8".into()))?;
    let mut lines = head_str.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| ParseError::Malformed("empty request".into()))?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| ParseError::Malformed("missing method".into()))?
        .to_ascii_uppercase();
    let target = parts
        .next()
        .ok_or_else(|| ParseError::Malformed("missing request target".into()))?;
    let version = parts
        .next()
        .ok_or_else(|| ParseError::Malformed("missing HTTP version".into()))?;
    if !version.starts_with("HTTP/1.") {
        return Err(ParseError::Malformed(format!(
            "unsupported HTTP version {version:?}"
        )));
    }
    let mut query = Vec::new();
    let path = match target.split_once('?') {
        Some((p, q)) => {
            for kv in q.split('&') {
                if let Some((k, v)) = kv.split_once('=') {
                    query.push((k.to_string(), v.to_string()));
                }
            }
            p.to_string()
        }
        None => target.to_string(),
    };

    let mut headers = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(ParseError::Malformed(format!("bad header line: {line:?}")));
        };
        let name = name.trim().to_ascii_lowercase();
        if name.is_empty() {
            return Err(ParseError::Malformed("empty header name".into()));
        }
        headers.push((name, value.trim().to_string()));
    }

    let content_length = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .map(|(_, v)| v.as_str());
    let body_start = end + 4;
    let body = match content_length {
        Some(v) => {
            let n: usize = v
                .parse()
                .map_err(|_| ParseError::Malformed(format!("bad content-length {v:?}")))?;
            if n > 8 * 1024 * 1024 {
                return Err(ParseError::Malformed("body too large".into()));
            }
            if buf.len() < body_start + n {
                return Ok(None); // body still arriving
            }
            buf[body_start..body_start + n].to_vec()
        }
        None => Vec::new(),
    };

    Ok(Some(Request {
        method,
        path,
        query,
        headers,
        body,
    }))
}

// (version/header validation errors are constructed inline above)

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// A response to write.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
}

impl Response {
    pub fn json(status: u16, value: &serde_json::Value) -> Self {
        Self {
            status,
            content_type: "application/json".into(),
            body: serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec()),
        }
    }

    pub fn text(status: u16, text: &str) -> Self {
        Self {
            status,
            content_type: "text/plain; charset=utf-8".into(),
            body: text.as_bytes().to_vec(),
        }
    }

    /// Serialize to wire bytes (status line + Content-Length always).
    pub fn to_bytes(&self) -> Vec<u8> {
        let reason = reason_phrase(self.status);
        let mut out = Vec::with_capacity(self.body.len() + 128);
        out.extend_from_slice(
            format!("HTTP/1.1 {} {}\r\n", self.status, reason).as_bytes(),
        );
        out.extend_from_slice(
            format!("Content-Type: {}\r\n", self.content_type).as_bytes(),
        );
        out.extend_from_slice(
            format!("Content-Length: {}\r\n", self.body.len()).as_bytes(),
        );
        out.extend_from_slice(b"Connection: keep-alive\r\n\r\n");
        out.extend_from_slice(&self.body);
        out
    }
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_get() {
        let raw = b"GET /v1/models?x=1 HTTP/1.1\r\nHost: localhost\r\n\r\n";
        let req = parse_request(raw).unwrap().unwrap();
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/v1/models");
        assert_eq!(req.query, vec![("x".to_string(), "1".to_string())]);
        assert_eq!(req.header("HOST"), Some("localhost"));
        assert!(req.body.is_empty());
    }

    #[test]
    fn parses_post_with_body() {
        let raw = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 13\r\n\r\n{\"model\":\"m\"}";
        let req = parse_request(raw).unwrap().unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.body, br#"{"model":"m"}"#);
        assert_eq!(req.content_length(), Some(13));
    }

    #[test]
    fn incomplete_until_body_arrives() {
        let raw = b"POST /v1/chat HTTP/1.1\r\nContent-Length: 5\r\n\r\nab";
        assert_eq!(parse_request(raw), Ok(None));
    }

    #[test]
    fn malformed_request_line() {
        let raw = b"NOT-A-REQUEST\r\n\r\n";
        match parse_request(raw) {
            Err(ParseError::Malformed(_)) => {}
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn bad_version_rejected() {
        let raw = b"GET / HTTP/3\r\n\r\n";
        assert!(matches!(parse_request(raw), Err(ParseError::Malformed(_))));
    }

    #[test]
    fn response_bytes_are_wellformed() {
        let r = Response::json(200, &serde_json::json!({"ok": true}));
        let bytes = r.to_bytes();
        let s = String::from_utf8(bytes).unwrap();
        assert!(s.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(s.contains("Content-Type: application/json\r\n"));
        assert!(s.ends_with(r#"{"ok":true}"#));
    }
}
