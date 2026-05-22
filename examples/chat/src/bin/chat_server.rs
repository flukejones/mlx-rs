//! OpenAI-compatible HTTP server. POST `/v1/chat/completions` with
//! either string or parts-array content (text + `image_url` data
//! URLs). `stream: true` returns SSE chunks.
//!
//! **MLX Metal stream thread-pin:** MLX binds its Metal stream to
//! the first thread that touches it. The server uses a current-
//! thread tokio runtime + `LocalSet` so `load` and every `generate`
//! call (including SSE delta callbacks via `spawn_local`) run on
//! the same OS thread.

#![allow(clippy::print_stderr, reason = "CLI binary logs to stderr")]
#![allow(clippy::print_stdout, reason = "CLI binary prints to stdout")]

use std::convert::Infallible;
use std::net::SocketAddr;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use argh::FromArgs;
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::post,
    Router,
};
use base64::Engine;
use chat::user_input::build_chat_input;
use mlx_lm::chat_template::{ChatMessage as LmChatMessage, ContentPart, MessageContent};
use mlx_lm::{
    generate, load, FinishReason, GenerateParams, Image as LmImage, ModelContext, SamplingParams,
    UserInput,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

const DEFAULT_PORT: u16 = 8080;
const DEFAULT_MAX_TOKENS: i32 = 512;
const MAX_BODY: usize = 32 * 1024 * 1024;
/// SSE channel depth. Exceeded → `try_send` drops chunks; stream
/// looks choppy. Dev-server default.
const SSE_CHANNEL_DEPTH: usize = 64;

/// OpenAI-compatible HTTP server for an mlx_lm checkpoint.
#[derive(FromArgs)]
struct Args {
    /// model directory
    #[argh(option)]
    model: PathBuf,

    /// listen port (default 8080)
    #[argh(option, default = "DEFAULT_PORT")]
    port: u16,
}

/// `generate` mutates KV cache; requests serialise through this Mutex.
type AppState = Arc<Mutex<ModelContext>>;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args: Args = argh::from_env();

    eprintln!("[loading {}]", args.model.display());
    let ctx = load(&args.model).context("load model")?;
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
    tokio::signal::ctrl_c().await.ok();
    eprintln!("[ctrl-c received, shutting down]");
}

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

struct ApiError(anyhow::Error);

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let msg = format!("{:#}", self.0);
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

    let input = build_chat_input(messages, images);
    let params = GenerateParams {
        max_new_tokens: max_tokens,
        sampling: SamplingParams { temperature, top_p },
        ..GenerateParams::default()
    };

    if stream {
        Ok(stream_response(state, input, params).into_response())
    } else {
        let result = {
            let mut ctx = state.lock().expect("ctx lock");
            generate(&mut ctx, input, params, &mut |_, _| {
                ControlFlow::Continue(())
            })
        }
        .context("generate")?;
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

fn stream_response(state: AppState, input: UserInput, params: GenerateParams) -> Response {
    let (tx, rx) = mpsc::channel::<axum::body::Bytes>(SSE_CHANNEL_DEPTH);
    let id: Arc<str> = Arc::from(chatcmpl_id());

    tokio::task::spawn_local(async move {
        send_chunk(
            &tx,
            ChatChunk {
                id: id.to_string(),
                object: "chat.completion.chunk",
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: Some("assistant"),
                        content: None,
                    },
                    finish_reason: None,
                }],
            },
        );

        let result = {
            let mut ctx = state.lock().expect("ctx lock");
            let id_token = id.clone();
            let tx_token = tx.clone();
            generate(&mut ctx, input, params, &mut |_, delta| {
                send_chunk(
                    &tx_token,
                    ChatChunk {
                        id: id_token.to_string(),
                        object: "chat.completion.chunk",
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: Some(delta.to_owned()),
                            },
                            finish_reason: None,
                        }],
                    },
                );
                ControlFlow::Continue(())
            })
        };

        let finish = match &result {
            Ok(r) => finish_reason_str(r.finish_reason),
            // generate errored — report length so clients don't treat
            // a transport failure as a normal completion.
            Err(_) => "length",
        };
        send_chunk(
            &tx,
            ChatChunk {
                id: id.to_string(),
                object: "chat.completion.chunk",
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta::default(),
                    finish_reason: Some(finish),
                }],
            },
        );
        send_done(&tx);

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

fn send_chunk(tx: &mpsc::Sender<axum::body::Bytes>, chunk: ChatChunk) {
    let payload = serde_json::to_string(&chunk).unwrap_or_default();
    let frame = format!("data: {payload}\n\n");
    tx.try_send(axum::body::Bytes::from(frame.into_bytes()))
        .ok();
}

fn send_done(tx: &mpsc::Sender<axum::body::Bytes>) {
    tx.try_send(axum::body::Bytes::from_static(b"data: [DONE]\n\n"))
        .ok();
}

/// Strip image parts out to a separate vec; rewrite each message's
/// content into the parts-list shape the chat template expects
/// (image parts followed by joined text).
fn into_user_input(messages: &[RequestMessage]) -> Result<(Vec<LmChatMessage>, Vec<LmImage>)> {
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
                            images.push(decode_image_url(&image_url.url)?);
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
            anyhow::bail!("http(s):// image URLs not supported; use data: URLs");
        }
        anyhow::bail!("unsupported image_url scheme: {url}");
    };
    let comma = rest
        .find(',')
        .ok_or_else(|| anyhow::anyhow!("malformed data URL: missing comma"))?;
    let header = &rest[..comma];
    let payload = &rest[comma + 1..];
    if !header.contains(";base64") {
        anyhow::bail!("data URL must use base64 encoding, got: {header}");
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.as_bytes())
        .context("base64 decode")?;
    let img = image::load_from_memory(&bytes).context("image decode")?;
    Ok(LmImage::Decoded(img))
}

fn finish_reason_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

fn chatcmpl_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-{nanos:032x}")
}
