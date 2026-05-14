//! Minimal OpenAI-compatible HTTP server for `chandra`.
//!
//! Implements POST `/v1/chat/completions` with the subset of the spec needed
//! by the retypst-ocr-chandra client:
//!
//! - `messages: [{role, content}]` where `content` may be a plain string or
//!   an array of `{type: "text", text}` / `{type: "image_url", image_url}`
//!   parts. The first image-url part is decoded (data-URL base64 or
//!   `http(s)://...`) and fed through the vision tower + multimodal
//!   stitcher; text parts of the last user message are concatenated for
//!   the prompt.
//! - `max_tokens`, `temperature`, `top_p` sampling controls.
//!
//! The response is non-streaming JSON of shape
//! `{ choices: [{ message: { role: "assistant", content: ... },
//! finish_reason }], usage: {...} }`.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Json, Response, Sse},
    routing::post,
    Router,
};
use tokio_stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;

#[allow(unused_imports)]
use crate::err;
use crate::{AppState, BoxError, Cli, FinishReason, Result};

pub fn run(cli: Cli) -> Result<()> {
    let state = Arc::new(AppState::from_dir(&cli.model)?);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async move { serve(cli.port, state).await })
}

async fn serve(port: u16, state: Arc<AppState>) -> Result<()> {
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("listening on {addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("ctrl-c received, shutting down");
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<i32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: ChatContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ChatContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(dead_code)]
enum ContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrl,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ImageUrl {
    url: String,
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct Choice {
    index: i32,
    message: ResponseMessage,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize)]
struct ResponseMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
}

struct ApiError(BoxError);

impl<E: Into<BoxError>> From<E> for ApiError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let msg = self.0.to_string();
        eprintln!("error: request failed: {msg}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": msg}})),
        )
            .into_response()
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    let user_text = extract_user_text(&req.messages)?;
    let image = first_image(&req.messages)?;
    let has_image = image.is_some();

    let rendered = state.render_chat_prompt(&user_text, has_image)?;
    let max_tokens = req.max_tokens.unwrap_or(512);
    let temperature = req.temperature.unwrap_or(0.0);
    let top_p = req.top_p;

    if req.stream {
        return Ok(stream_chat_completion(
            state, image, rendered, max_tokens, temperature, top_p,
        )
        .into_response());
    }

    let result = match image {
        Some(img) => state.generate_multimodal(&rendered, img, max_tokens, temperature, top_p)?,
        None => state.generate_text(&rendered, max_tokens, temperature, top_p)?,
    };

    Ok(Json(ChatResponse {
        id: format!("chatcmpl-{}", uuid_like()),
        object: "chat.completion",
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant",
                content: result.text,
            },
            finish_reason: result.finish_reason,
        }],
        usage: Usage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            total_tokens: result.prompt_tokens + result.completion_tokens,
        },
    })
    .into_response())
}

fn stream_chat_completion(
    state: Arc<AppState>,
    image: Option<image::DynamicImage>,
    rendered: String,
    max_tokens: i32,
    temperature: f32,
    top_p: Option<f32>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    let id = format!("chatcmpl-{}", uuid_like());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // First chunk: role marker.
    let role_chunk = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "chandra",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": null,
        }],
    });
    let _ = tx.try_send(Event::default().data(role_chunk.to_string()));

    let tx_clone = tx.clone();
    let id_clone = id.clone();
    std::thread::spawn(move || {
        let result = match image {
            Some(img) => state.stream_multimodal(
                &rendered,
                img,
                max_tokens,
                temperature,
                top_p,
                |_tok, delta| {
                    if delta.is_empty() {
                        return std::ops::ControlFlow::Continue(());
                    }
                    let chunk = serde_json::json!({
                        "id": id_clone,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "chandra",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": null,
                        }],
                    });
                    if tx_clone
                        .blocking_send(Event::default().data(chunk.to_string()))
                        .is_err()
                    {
                        return std::ops::ControlFlow::Break(());
                    }
                    std::ops::ControlFlow::Continue(())
                },
            ),
            None => state.stream_text(
                &rendered,
                max_tokens,
                temperature,
                top_p,
                |_tok, delta| {
                    if delta.is_empty() {
                        return std::ops::ControlFlow::Continue(());
                    }
                    let chunk = serde_json::json!({
                        "id": id_clone,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "chandra",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": null,
                        }],
                    });
                    if tx_clone
                        .blocking_send(Event::default().data(chunk.to_string()))
                        .is_err()
                    {
                        return std::ops::ControlFlow::Break(());
                    }
                    std::ops::ControlFlow::Continue(())
                },
            ),
        };

        let finish_reason = match &result {
            Ok(r) => match r.finish_reason {
                FinishReason::Stop => "stop",
                FinishReason::Length => "length",
            },
            Err(_) => "stop",
        };
        let final_chunk = serde_json::json!({
            "id": id_clone,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "chandra",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }],
        });
        let _ = tx_clone.blocking_send(Event::default().data(final_chunk.to_string()));
        let _ = tx_clone.blocking_send(Event::default().data("[DONE]"));
        if let Err(e) = result {
            eprintln!("error: streaming generation failed: {e:#}");
        }
    });

    Sse::new(ReceiverStream::new(rx).map(Ok))
}

fn extract_user_text(messages: &[ChatMessage]) -> Result<String> {
    let last = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .ok_or_else(|| err!("no user message in `messages`"))?;
    let mut buf = String::new();
    match &last.content {
        ChatContent::Text(s) => buf.push_str(s),
        ChatContent::Parts(parts) => {
            for part in parts {
                if let ContentPart::Text { text } = part {
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str(text);
                }
            }
        }
    }
    Ok(buf)
}

fn first_image(messages: &[ChatMessage]) -> Result<Option<image::DynamicImage>> {
    use base64::Engine;
    for m in messages {
        if let ChatContent::Parts(parts) = &m.content {
            for part in parts {
                if let ContentPart::ImageUrl { image_url } = part {
                    let url = image_url.url.trim();
                    if let Some(rest) = url.strip_prefix("data:") {
                        // data:image/png;base64,...
                        let comma = rest
                            .find(',')
                            .ok_or_else(|| err!("malformed data URL: missing comma"))?;
                        let header = &rest[..comma];
                        let payload = &rest[comma + 1..];
                        if !header.contains(";base64") {
                            return Err(err!(
                                "data URL must use base64 encoding, got: {header}"
                            ));
                        }
                        let bytes = base64::engine::general_purpose::STANDARD
                            .decode(payload.as_bytes())
                            .map_err(|e| err!("base64 decode: {e}"))?;
                        let img = image::load_from_memory(&bytes)
                            .map_err(|e| err!("image decode: {e}"))?;
                        return Ok(Some(img));
                    } else if url.starts_with("http://") || url.starts_with("https://") {
                        return Err(err!(
                            "http(s):// image URLs not supported by this example; use data: URLs"
                        ));
                    } else {
                        return Err(err!(
                            "unsupported image_url scheme: {url}"
                        ));
                    }
                }
            }
        }
    }
    Ok(None)
}

fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:032x}")
}
