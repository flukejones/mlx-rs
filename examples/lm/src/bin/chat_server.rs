//! OpenAI-compatible HTTP server for any `mlx_lm` checkpoint.
//!
//! POST `/v1/chat/completions` with the subset of the OpenAI spec
//! needed for chat + vision-LM clients:
//!
//! - `messages: [{role, content}]` where `content` is a plain string
//!   *or* a parts array of `{type:"text",text}` /
//!   `{type:"image_url",image_url:{url}}` objects. Image URLs accept
//!   `data:` base64 inline data.
//! - `max_tokens`, `temperature`, `top_p` sampling controls.
//! - `stream: true` returns an SSE stream of OpenAI-flavoured chunks
//!   (`data: {...}\n\n` per delta).
//!
//! Non-streaming responses are plain JSON
//! `{ choices: [{ message: { role:"assistant", content }, finish_reason }],
//!    usage: { prompt_tokens, completion_tokens, total_tokens } }`.
//!
//! **MLX Metal stream thread-pin:** MLX initialises its Metal stream
//! on whichever thread first touched it. The server is built on a
//! `tokio::runtime::Builder::new_current_thread()` + `LocalSet`, and
//! every `generate` call runs inline on that thread (the same one
//! the model was loaded on). The SSE writer pumps deltas through a
//! `tokio::sync::mpsc::Sender` drained by `tokio::spawn_local` on
//! the same `LocalSet`, so the `on_token` callback never crosses a
//! thread boundary.

#![allow(clippy::print_stderr, reason = "CLI binary logs to stderr")]
#![allow(clippy::print_stdout, reason = "CLI binary prints to stdout")]

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::post,
    Router,
};
use base64::Engine;
use mlx_lm::chat_template::{ChatMessage as LmChatMessage, ContentPart, MessageContent};
use mlx_lm::{
    generate, load, FinishReason, GenerateParams, Image as LmImage, ModelContext, Prompt,
    SamplingParams, UserInput,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

type BoxError = Box<dyn std::error::Error + Send + Sync>;
type Result<T> = std::result::Result<T, BoxError>;

const DEFAULT_PORT: u16 = 8080;
const DEFAULT_MAX_TOKENS: i32 = 512;
const MAX_BODY: usize = 32 * 1024 * 1024;

struct Args {
    model: PathBuf,
    port: u16,
}

fn parse_args() -> Result<Args> {
    let mut model: Option<PathBuf> = None;
    let mut port: u16 = DEFAULT_PORT;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(it.next().ok_or("--model needs a path")?)),
            "--port" => port = it.next().ok_or("--port needs a value")?.parse()?,
            "-h" | "--help" => {
                println!("chat_server --model <dir> [--port {DEFAULT_PORT}]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        port,
    })
}

/// Boxed context behind a Mutex — `generate` mutates internal cache
/// state, so requests serialise.
type AppState = Arc<Mutex<ModelContext>>;

fn main() -> Result<()> {
    let args = parse_args()?;
    eprintln!("[loading {}]", args.model.display());
    let ctx = load(&args.model)?;
    eprintln!("[loaded]");
    let state: AppState = Arc::new(Mutex::new(ctx));

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async move { serve(args.port, state).await })
}

async fn serve(port: u16, state: AppState) -> Result<()> {
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(state);
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("[listening on {addr}]");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("[ctrl-c received, shutting down]");
}

// ─── OpenAI request shape ───────────────────────────────────────

#[derive(Debug, Deserialize)]
#[allow(dead_code, reason = "OpenAI request fields parsed but not all used")]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<RequestMessage>,
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
struct RequestMessage {
    role: String,
    content: RequestContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RequestContent {
    Text(String),
    Parts(Vec<RequestPart>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RequestPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct ImageUrl {
    url: String,
}

// ─── OpenAI response shape ──────────────────────────────────────

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
    finish_reason: &'static str,
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

// ─── Streaming chunk shape ──────────────────────────────────────

#[derive(Debug, Serialize)]
struct ChatChunk {
    id: String,
    object: &'static str,
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
struct ChunkChoice {
    index: i32,
    delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize, Default)]
struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

// ─── Route handler ──────────────────────────────────────────────

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
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> std::result::Result<Response, ApiError> {
    let (messages, images) = into_user_input(&req.messages)?;
    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let temperature = req.temperature.unwrap_or(0.0);
    let top_p = req.top_p;
    let stream = req.stream;

    let input = build_input(messages, images);

    let params = GenerateParams {
        max_new_tokens: max_tokens,
        sampling: SamplingParams { temperature, top_p },
        extra_stop_ids: Vec::new(),
    };

    if stream {
        Ok(stream_response(state, input, params).into_response())
    } else {
        let result = {
            let mut ctx = state.lock().expect("ctx lock");
            generate(
                &mut ctx,
                input,
                params,
                &mut |_, _| std::ops::ControlFlow::Continue(()),
            )
        }
        .map_err(|e| format!("generate: {e}"))?;
        Ok(Json(ChatResponse {
            id: chatcmpl_id(),
            object: "chat.completion",
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant",
                    content: result.text,
                },
                finish_reason: finish_reason_str(result.finish_reason),
            }],
            usage: Usage {
                prompt_tokens: result.prompt_tokens,
                completion_tokens: result.completion_tokens,
                total_tokens: result.prompt_tokens + result.completion_tokens,
            },
        })
        .into_response())
    }
}

fn build_input(messages: Vec<LmChatMessage>, images: Vec<LmImage>) -> UserInput {
    UserInput {
        prompt: Prompt::Chat(messages),
        images,
        audios: Vec::new(),
        videos: Vec::new(),
    }
}

/// Spawn the generate call on the LocalSet (same thread as the
/// model). The `on_token` callback pushes formatted SSE frames into
/// an `mpsc::Sender`; the receiver is wrapped in a stream that
/// becomes the HTTP response body.
fn stream_response(state: AppState, input: UserInput, params: GenerateParams) -> Response {
    let (tx, rx) = mpsc::channel::<axum::body::Bytes>(64);
    let id = Rc::new(chatcmpl_id());
    let id_for_task = id.clone();

    tokio::task::spawn_local(async move {
        let send_frame = |frame: String| {
            let bytes = axum::body::Bytes::from(frame.into_bytes());
            // `try_send` drops a chunk if the receiver is more than
            // 64 frames behind; the stream then appears choppy but
            // generation continues. Fine for a dev server.
            let _ = tx.try_send(bytes);
        };

        let header = ChatChunk {
            id: (*id_for_task).clone(),
            object: "chat.completion.chunk",
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant"),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        send_frame(format!(
            "data: {}\n\n",
            serde_json::to_string(&header).unwrap_or_default()
        ));

        let result = {
            let mut ctx = state.lock().expect("ctx lock");
            let id_inner = id_for_task.clone();
            let tx_inner = tx.clone();
            generate(&mut ctx, input, params, &mut |_, delta| {
                let chunk = ChatChunk {
                    id: (*id_inner).clone(),
                    object: "chat.completion.chunk",
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: None,
                            content: Some(delta.to_owned()),
                        },
                        finish_reason: None,
                    }],
                };
                let frame = format!(
                    "data: {}\n\n",
                    serde_json::to_string(&chunk).unwrap_or_default()
                );
                let _ = tx_inner.try_send(axum::body::Bytes::from(frame.into_bytes()));
                std::ops::ControlFlow::Continue(())
            })
        };

        let finish = match &result {
            Ok(r) => finish_reason_str(r.finish_reason),
            Err(_) => "stop",
        };
        let closing = ChatChunk {
            id: (*id_for_task).clone(),
            object: "chat.completion.chunk",
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta::default(),
                finish_reason: Some(finish),
            }],
        };
        send_frame(format!(
            "data: {}\n\n",
            serde_json::to_string(&closing).unwrap_or_default()
        ));
        send_frame("data: [DONE]\n\n".to_owned());

        if let Err(e) = result {
            eprintln!("error: stream generate: {e}");
        }
    });

    let stream = ReceiverStream::new(rx).map(std::result::Result::<_, Infallible>::Ok);

    Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(stream))
        .expect("build sse response")
}

/// Walk the OpenAI request messages: strip image parts out to the
/// `images` vec, replace the message's content with the parts-list
/// shape the chat template expects (one `ContentPart::Image` per
/// attached image, followed by the text). Plain-text content stays
/// as-is.
fn into_user_input(
    messages: &[RequestMessage],
) -> Result<(Vec<LmChatMessage>, Vec<LmImage>)> {
    let mut out_msgs: Vec<LmChatMessage> = Vec::with_capacity(messages.len());
    let mut images: Vec<LmImage> = Vec::new();
    for m in messages {
        let role = m.role.clone();
        match &m.content {
            RequestContent::Text(s) => {
                out_msgs.push(LmChatMessage {
                    role,
                    content: MessageContent::Text(s.clone()),
                });
            }
            RequestContent::Parts(parts) => {
                let mut text_chunks: Vec<String> = Vec::new();
                let mut msg_image_count = 0_usize;
                for part in parts {
                    match part {
                        RequestPart::Text { text } => text_chunks.push(text.clone()),
                        RequestPart::ImageUrl { image_url } => {
                            msg_image_count += 1;
                            let img = decode_image_url(&image_url.url)?;
                            images.push(img);
                        }
                        RequestPart::Unknown => {}
                    }
                }
                let joined = text_chunks.join("\n");
                let content = if msg_image_count > 0 {
                    let mut new_parts: Vec<ContentPart> =
                        (0..msg_image_count).map(|_| ContentPart::Image).collect();
                    new_parts.push(ContentPart::Text { text: joined });
                    MessageContent::Parts(new_parts)
                } else {
                    MessageContent::Text(joined)
                };
                out_msgs.push(LmChatMessage { role, content });
            }
        }
    }
    Ok((out_msgs, images))
}

fn decode_image_url(url: &str) -> Result<LmImage> {
    let url = url.trim();
    let Some(rest) = url.strip_prefix("data:") else {
        if url.starts_with("http://") || url.starts_with("https://") {
            return Err("http(s):// image URLs not supported; use data: URLs".into());
        }
        return Err(format!("unsupported image_url scheme: {url}").into());
    };
    let comma = rest.find(',').ok_or("malformed data URL: missing comma")?;
    let header = &rest[..comma];
    let payload = &rest[comma + 1..];
    if !header.contains(";base64") {
        return Err(format!("data URL must use base64 encoding, got: {header}").into());
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.as_bytes())
        .map_err(|e| format!("base64 decode: {e}"))?;
    let img = image::load_from_memory(&bytes).map_err(|e| format!("image decode: {e}"))?;
    Ok(LmImage::Decoded(img))
}

fn finish_reason_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

fn chatcmpl_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-{nanos:032x}")
}
