//! Jinja chat-template rendering.
//!
//! Mirrors HF `tokenizer.apply_chat_template`: load
//! `chat_template.jinja` (preferred) or
//! `tokenizer_config.json::chat_template` (fallback), then render
//! `ChatMessage`s through it. `MessageContent` covers both the
//! plain-string form (llama-family) and the parts-list form
//! (qwen3-vl / qwen3.5 multimodal).

use std::path::Path;

use minijinja::{context, Environment, Value};
use serde::{Deserialize, Serialize};

use crate::error::Error;

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `"system"`, `"user"`, `"assistant"`, or `"tool"`.
    pub role: String,
    /// String or list-of-parts content.
    pub content: MessageContent,
}

impl ChatMessage {
    /// Build a user message with plain-text content.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: MessageContent::Text(text.into()),
        }
    }

    /// User message: `[image, text]`. Vision model splices the image
    /// into the `<|image_pad|>` slot at runtime.
    pub fn user_with_image(text: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart::Image,
                ContentPart::Text { text: text.into() },
            ]),
        }
    }

    /// Build a system message with plain-text content.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: MessageContent::Text(text.into()),
        }
    }

    /// Build an assistant message with plain-text content.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: MessageContent::Text(text.into()),
        }
    }
}

/// Plain string (llama-family) or typed-parts list (qwen3-vl,
/// qwen3.5 — needed for image / tool-call messages).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// List of typed parts (text + image, etc.).
    Parts(Vec<ContentPart>),
}

/// One element of a parts-list message content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentPart {
    /// Plain text part.
    Text {
        /// The text content.
        text: String,
    },
    /// Image placeholder. Template emits image-pad token(s);
    /// vision features replace them at runtime.
    Image,
}

/// Holder for a parsed chat template, ready to render messages.
pub struct ChatTemplate {
    source: String,
}

impl ChatTemplate {
    /// Load the chat template from a checkpoint dir. Looks for
    /// `chat_template.jinja` first, then `tokenizer_config.json::chat_template`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let jinja = dir.join("chat_template.jinja");
        if jinja.exists() {
            let source = std::fs::read_to_string(&jinja)?;
            return Ok(Self { source });
        }
        let tokcfg_path = dir.join("tokenizer_config.json");
        let raw = std::fs::read_to_string(&tokcfg_path)?;
        let parsed: serde_json::Value = serde_json::from_str(&raw)?;
        let source = parsed
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                Error::Other(
                    format!("no chat_template at {} or {tokcfg_path:?}", jinja.display()).into(),
                )
            })?.to_owned();
        Ok(Self { source })
    }

    /// Build from a raw template string (testing / non-standard sources).
    pub fn from_source(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
        }
    }

    /// Render `messages` through the template.
    /// `add_generation_prompt`: true for inference (appends
    /// assistant turn-start), false for fine-tune data prep.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, Error> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("chat", &self.source)
            .map_err(|e| Error::Other(format!("compiling chat template: {e}").into()))?;
        let tmpl = env
            .get_template("chat")
            .map_err(|e| Error::Other(format!("loading chat template: {e}").into()))?;
        let messages_value = Value::from_serialize(messages);
        let ctx = context! {
            messages => messages_value,
            add_generation_prompt => add_generation_prompt,
        };
        tmpl.render(ctx)
            .map_err(|e| Error::Other(format!("rendering chat template: {e}").into()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use super::*;

    /// A simple template that just emits `role: content` per message.
    /// Uses explicit `|` separators instead of newlines to keep the
    /// whitespace-control rules predictable in tests.
    const TINY_TMPL: &str = "\
        {% for m in messages %}\
        {{ m.role }}={% if m.content is string %}{{ m.content }}\
        {% else %}\
        {% for p in m.content %}<{{ p.type }}>{% if p.type == 'text' %}{{ p.text }}{% endif %}{% endfor %}\
        {% endif %}|\
        {% endfor %}\
        {% if add_generation_prompt %}assistant={% endif %}";

    #[test]
    fn renders_plain_user_message() {
        let tmpl = ChatTemplate::from_source(TINY_TMPL);
        let out = tmpl.render(&[ChatMessage::user("Hello")], true).unwrap();
        assert!(out.contains("user=Hello"), "got: {out}");
        assert!(out.ends_with("assistant="), "got: {out}");
    }

    #[test]
    fn renders_parts_list_user_message() {
        let tmpl = ChatTemplate::from_source(TINY_TMPL);
        let out = tmpl
            .render(&[ChatMessage::user_with_image("What is this?")], true)
            .unwrap();
        assert!(
            out.contains("user=<image><text>What is this?"),
            "got: {out}"
        );
    }

    #[test]
    fn renders_system_user_assistant_turn() {
        let tmpl = ChatTemplate::from_source(TINY_TMPL);
        let out = tmpl
            .render(
                &[
                    ChatMessage::system("be helpful"),
                    ChatMessage::user("Hi"),
                    ChatMessage::assistant("Hello!"),
                    ChatMessage::user("Again"),
                ],
                true,
            )
            .unwrap();
        assert!(out.contains("system=be helpful"), "got: {out}");
        assert!(out.contains("user=Hi"), "got: {out}");
        assert!(out.contains("assistant=Hello!"), "got: {out}");
        assert!(out.contains("user=Again"), "got: {out}");
        assert!(out.ends_with("assistant="), "got: {out}");
    }
}
