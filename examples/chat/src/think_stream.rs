//! Pre-filter that strips `<think>` / `</think>` tags before
//! forwarding to a [`MarkdownStreamer`]. Reasoning content streams
//! through unchanged; only the tag literals are suppressed.

use std::io::{self, IsTerminal, Write};

use crate::markdown_stream::MarkdownStreamer;

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// Wraps a [`MarkdownStreamer`]; strips `<think>` / `</think>` tags
/// from streamed deltas before forwarding.
pub struct ThinkStream<W: Write> {
    inner: MarkdownStreamer<W>,
    /// Holds only the trailing bytes that could be a tag prefix
    /// straddling the next push; everything else flushes to `inner`.
    buf: String,
}

impl<W: Write> ThinkStream<W> {
    pub fn new(out: W) -> Self
    where
        W: IsTerminal,
    {
        Self {
            inner: MarkdownStreamer::new(out),
            buf: String::new(),
        }
    }

    /// Build with an explicit TTY flag; mirrors
    /// [`MarkdownStreamer::with_tty`] for tests.
    pub fn with_tty(out: W, tty: bool) -> Self {
        Self {
            inner: MarkdownStreamer::with_tty(out, tty),
            buf: String::new(),
        }
    }

    pub fn push(&mut self, delta: &str) -> io::Result<()> {
        self.buf.push_str(delta);
        loop {
            let next = first_tag(&self.buf);
            match next {
                Some((pos, len)) => {
                    let before = self.buf[..pos].to_owned();
                    let after_start = pos + len;
                    let after = self.buf[after_start..].to_owned();
                    if !before.is_empty() {
                        self.inner.push(&before)?;
                    }
                    self.buf = after;
                }
                None => {
                    let hold = tail_partial_tag_len(&self.buf, THINK_OPEN)
                        .max(tail_partial_tag_len(&self.buf, THINK_CLOSE));
                    let split = self.buf.len() - hold;
                    if split > 0 {
                        let flushable = self.buf[..split].to_owned();
                        self.buf.drain(..split);
                        self.inner.push(&flushable)?;
                    }
                    return Ok(());
                }
            }
        }
    }

    pub fn finish(&mut self) -> io::Result<()> {
        if !self.buf.is_empty() {
            let trailing = std::mem::take(&mut self.buf);
            self.inner.push(&trailing)?;
        }
        self.inner.finish()
    }

    pub fn into_inner(self) -> W {
        self.inner.into_inner()
    }
}

/// `(byte_pos, byte_len)` of the earliest `<think>` / `</think>`.
fn first_tag(buf: &str) -> Option<(usize, usize)> {
    let open = buf.find(THINK_OPEN).map(|p| (p, THINK_OPEN.len()));
    let close = buf.find(THINK_CLOSE).map(|p| (p, THINK_CLOSE.len()));
    match (open, close) {
        (Some(a), Some(b)) if a.0 <= b.0 => Some(a),
        (Some(_), Some(b)) => Some(b),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Length of the longest tail of `buf` that is a strict prefix of
/// `needle`. The streamer holds back exactly that many chars between
/// pushes so a split tag can still match.
fn tail_partial_tag_len(buf: &str, needle: &str) -> usize {
    let max = (needle.len() - 1).min(buf.len());
    for k in (1..=max).rev() {
        if buf.is_char_boundary(buf.len() - k) && needle.starts_with(&buf[buf.len() - k..]) {
            return k;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(input: &str) -> String {
        let mut s = ThinkStream::with_tty(Vec::<u8>::new(), false);
        s.push(input).unwrap();
        s.finish().unwrap();
        String::from_utf8(s.into_inner()).unwrap()
    }

    fn render_split(chunks: &[&str]) -> String {
        let mut s = ThinkStream::with_tty(Vec::<u8>::new(), false);
        for c in chunks {
            s.push(c).unwrap();
        }
        s.finish().unwrap();
        String::from_utf8(s.into_inner()).unwrap()
    }

    #[test]
    fn passthrough_when_no_tags() {
        assert_eq!(render("plain text\n"), "plain text\n");
    }

    #[test]
    fn strips_open_and_close_pair() {
        assert_eq!(
            render("hi <think>reason</think> after\n"),
            "hi reason after\n"
        );
    }

    #[test]
    fn strips_stray_close_only() {
        // qwen 3.6 pattern: only `</think>` arrives; opener was in the prompt.
        assert_eq!(render("reason</think>answer\n"), "reasonanswer\n");
    }

    #[test]
    fn strips_stray_open_only() {
        assert_eq!(render("<think>only opener"), "only opener");
    }

    #[test]
    fn tag_split_across_pushes() {
        assert_eq!(
            render_split(&["before <thi", "nk>reason</thi", "nk> after\n"]),
            "before reason after\n"
        );
    }

    #[test]
    fn tag_at_very_start() {
        assert_eq!(render("</think>plain text\n"), "plain text\n");
    }

    #[test]
    fn no_false_positive_on_partial_tag_at_eof() {
        // `finish()` flushes a held-back tag prefix as raw content.
        assert_eq!(render("text <thi"), "text <thi");
    }
}
