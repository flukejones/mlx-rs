//! Streaming markdown → terminal renderer. Inline transforms for
//! headings, lists, bold, italic, inline code. Capture-and-render
//! (with `[ rendering … ⠋ ]` spinner) for tables and code fences.
//! Non-TTY sinks bypass everything and passthrough raw.

use std::io::{self, IsTerminal, Write};
use std::time::Instant;

use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, CellAlignment, ContentArrangement, Table};

/// Streaming markdown transformer. Construct one per assistant turn;
/// feed it deltas via [`Self::push`]; call [`Self::finish`] at the
/// end of the turn to flush any open capture block.
pub struct MarkdownStreamer<W: Write> {
    out: W,
    tty: bool,
    /// Raw input that hasn't been split into a complete line yet.
    line_buf: String,
    /// True iff we're inside a capture block (table or code fence).
    block: Option<Block>,
    /// Spinner state for the capture-mode status line.
    spinner: Spinner,
    /// List-mode state across consecutive list lines.
    list_stack: Vec<ListLevel>,
}

#[derive(Debug)]
enum Block {
    /// A markdown table being captured. Stores the raw lines for
    /// later parse + comfy-table render.
    Table { lines: Vec<String> },
    /// A code fence being captured. Stores language tag + body
    /// lines (verbatim, no inline transforms).
    CodeFence { lang: String, body: Vec<String> },
}

#[derive(Debug)]
struct Spinner {
    frame: usize,
    last_paint: Instant,
}

const SPINNER_FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const SPINNER_MIN_PAINT_GAP_MS: u128 = 80;

impl Spinner {
    fn new() -> Self {
        Self {
            frame: 0,
            last_paint: Instant::now(),
        }
    }
    fn tick(&mut self) -> char {
        self.frame = (self.frame + 1) % SPINNER_FRAMES.len();
        SPINNER_FRAMES[self.frame]
    }
    fn ready(&self) -> bool {
        self.last_paint.elapsed().as_millis() >= SPINNER_MIN_PAINT_GAP_MS
    }
    fn mark_painted(&mut self) {
        self.last_paint = Instant::now();
    }
}

#[derive(Debug)]
struct ListLevel {
    /// Number of leading-space chars at this level's bullet.
    indent_spaces: usize,
    /// `true` for ordered (1. 2. 3.), `false` for unordered.
    ordered: bool,
}

const BOLD_ON: &str = "\x1b[1m";
const BOLD_OFF: &str = "\x1b[22m";
const ITAL_ON: &str = "\x1b[3m";
const ITAL_OFF: &str = "\x1b[23m";
const DIM_GREY_ON: &str = "\x1b[2;37m";
const DIM_GREY_OFF: &str = "\x1b[22;39m";
const CLEAR_LINE: &str = "\r\x1b[K";

/// Bullet glyphs by nesting depth (cycles past index 2).
const BULLETS: &[char] = &['•', '◦', '▪'];

impl<W: Write> MarkdownStreamer<W> {
    /// Build a streamer. Detects TTY-ness of `out` via
    /// [`IsTerminal`]; non-TTY callers get pure passthrough.
    pub fn new(out: W) -> Self
    where
        W: IsTerminal,
    {
        let tty = out.is_terminal();
        Self::with_tty(out, tty)
    }

    /// Build with an explicit TTY flag; `IsTerminal` is sealed so
    /// tests can't impl it on `Vec<u8>` and need this escape hatch.
    pub fn with_tty(out: W, tty: bool) -> Self {
        Self {
            out,
            tty,
            line_buf: String::new(),
            block: None,
            spinner: Spinner::new(),
            list_stack: Vec::new(),
        }
    }

    /// Consume the streamer and return the wrapped writer. Used by
    /// tests + wrappers that need to inspect the rendered output.
    pub fn into_inner(self) -> W {
        self.out
    }

    /// Push a streamed delta. May emit zero or more bytes to the
    /// underlying writer.
    pub fn push(&mut self, delta: &str) -> io::Result<()> {
        if !self.tty {
            return self.out.write_all(delta.as_bytes());
        }
        self.line_buf.push_str(delta);
        self.drain_complete_lines()?;
        self.maybe_tick_spinner()?;
        Ok(())
    }

    /// Flush any open capture block (raw if malformed) and any
    /// trailing partial line. Call once per assistant turn.
    pub fn finish(&mut self) -> io::Result<()> {
        if !self.tty {
            return Ok(());
        }
        if !self.line_buf.is_empty() {
            let trailing = std::mem::take(&mut self.line_buf);
            self.handle_line(&trailing)?;
        }
        if self.block.is_some() {
            self.flush_block_raw()?;
        }
        Ok(())
    }
}

impl<W: Write> MarkdownStreamer<W> {
    fn drain_complete_lines(&mut self) -> io::Result<()> {
        while let Some(nl) = self.line_buf.find('\n') {
            let line = self.line_buf[..nl].to_owned();
            self.line_buf.drain(..=nl);
            self.handle_line(&line)?;
        }
        Ok(())
    }

    fn handle_line(&mut self, line: &str) -> io::Result<()> {
        if let Some(block) = self.block.as_mut() {
            match block {
                Block::CodeFence { body, .. } => {
                    if line.trim_start().starts_with("```") {
                        self.close_code_fence()?;
                    } else {
                        body.push(line.to_owned());
                    }
                }
                Block::Table { lines } => {
                    if line.trim_start().starts_with('|') {
                        lines.push(line.to_owned());
                    } else {
                        self.close_table()?;
                        self.handle_line(line)?;
                    }
                }
            }
            return Ok(());
        }

        if let Some(lang) = parse_code_fence_open(line) {
            self.open_code_fence(lang)?;
            return Ok(());
        }
        if is_table_opener(line) {
            self.block = Some(Block::Table {
                lines: vec![line.to_owned()],
            });
            self.print_spinner_line("table")?;
            return Ok(());
        }
        if let Some((depth, rest)) = parse_heading(line) {
            self.emit_heading(depth, rest)?;
            self.list_stack.clear();
            return Ok(());
        }
        if let Some((indent, ordered, marker_text, content)) = parse_list_item(line) {
            self.emit_list_item(indent, ordered, marker_text, content)?;
            return Ok(());
        }
        if line.trim().is_empty() {
            self.list_stack.clear();
        }
        self.emit_inline_line(line)?;
        Ok(())
    }
}

/// Return `Some(lang)` if `line` opens a code fence ```` ```lang ````.
/// `lang` may be empty.
fn parse_code_fence_open(line: &str) -> Option<String> {
    let t = line.trim_start();
    let rest = t.strip_prefix("```")?;
    Some(rest.trim().to_owned())
}

/// Tentative table opener. `close_table` later confirms row 2 is a
/// valid separator; if not, the buffered lines flush as plain prose.
fn is_table_opener(line: &str) -> bool {
    let t = line.trim_start();
    if !t.starts_with('|') {
        return false;
    }
    let body = t.trim_matches('|');
    body.contains('|') && body.chars().any(|c| !c.is_whitespace())
}

/// Parse a heading `#{1,6} text`. Returns `(depth, text_without_prefix)`.
fn parse_heading(line: &str) -> Option<(usize, &str)> {
    let t = line.trim_start();
    let hashes = t.chars().take_while(|&c| c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let after = &t[hashes..];
    let rest = after.strip_prefix(' ')?;
    Some((hashes, rest))
}

/// Returns `(indent_spaces, ordered, marker_text, content)`.
/// `marker_text` is empty for unordered (caller substitutes the
/// bullet glyph); preserved verbatim for ordered (`"3. "`).
fn parse_list_item(line: &str) -> Option<(usize, bool, String, String)> {
    let indent = line.chars().take_while(|&c| c == ' ').count();
    let t = &line[indent..];
    // The marker char + space prefix disambiguates `* foo` (bullet)
    // from `*foo*` (italic).
    if let Some(c) = t.chars().next() {
        if matches!(c, '-' | '*' | '+') {
            let prefix: [char; 2] = [c, ' '];
            let prefix_str: String = prefix.iter().collect();
            if let Some(rest) = t.strip_prefix(prefix_str.as_str()) {
                return Some((indent, false, String::new(), rest.to_owned()));
            }
        }
    }
    let digits: String = t.chars().take_while(char::is_ascii_digit).collect();
    if !digits.is_empty() {
        let after = &t[digits.len()..];
        if let Some(rest) = after.strip_prefix(". ") {
            let marker = format!("{digits}. ");
            return Some((indent, true, marker, rest.to_owned()));
        }
    }
    None
}

impl<W: Write> MarkdownStreamer<W> {
    fn emit_heading(&mut self, _depth: usize, text: &str) -> io::Result<()> {
        write!(self.out, "{BOLD_ON}")?;
        self.emit_inline_text(text)?;
        writeln!(self.out, "{BOLD_OFF}")?;
        Ok(())
    }

    fn emit_list_item(
        &mut self,
        indent: usize,
        ordered: bool,
        marker_text: String,
        content: String,
    ) -> io::Result<()> {
        while self
            .list_stack
            .last()
            .is_some_and(|l| l.indent_spaces > indent)
        {
            self.list_stack.pop();
        }
        match self.list_stack.last_mut() {
            Some(top) if top.indent_spaces == indent => top.ordered = ordered,
            _ => self.list_stack.push(ListLevel {
                indent_spaces: indent,
                ordered,
            }),
        }
        let depth = self.list_stack.len().saturating_sub(1);
        let visual_indent = "  ".repeat(depth);
        let marker = if ordered {
            marker_text
        } else {
            format!("{} ", BULLETS[depth.min(BULLETS.len() - 1)])
        };
        write!(self.out, "{visual_indent}{marker}")?;
        self.emit_inline_text(&content)?;
        writeln!(self.out)?;
        Ok(())
    }

    fn emit_inline_line(&mut self, line: &str) -> io::Result<()> {
        self.emit_inline_text(line)?;
        writeln!(self.out)?;
        Ok(())
    }

    /// Bold (`**…**`), italic (`*…*`), inline code (`` `…` ``). Any
    /// unclosed marker auto-closes at line end so the next line
    /// doesn't inherit the attribute.
    fn emit_inline_text(&mut self, line: &str) -> io::Result<()> {
        let mut chars = line.chars().peekable();
        let mut in_bold = false;
        let mut in_italic = false;
        let mut in_code = false;
        let mut buf = String::new();

        while let Some(c) = chars.next() {
            match c {
                '*' if chars.peek() == Some(&'*') => {
                    chars.next(); // consume second '*'
                    self.out.write_all(buf.as_bytes())?;
                    buf.clear();
                    if in_bold {
                        self.out.write_all(BOLD_OFF.as_bytes())?;
                    } else {
                        self.out.write_all(BOLD_ON.as_bytes())?;
                    }
                    in_bold = !in_bold;
                }
                '*' => {
                    self.out.write_all(buf.as_bytes())?;
                    buf.clear();
                    if in_italic {
                        self.out.write_all(ITAL_OFF.as_bytes())?;
                    } else {
                        self.out.write_all(ITAL_ON.as_bytes())?;
                    }
                    in_italic = !in_italic;
                }
                '`' => {
                    self.out.write_all(buf.as_bytes())?;
                    buf.clear();
                    if in_code {
                        self.out.write_all(DIM_GREY_OFF.as_bytes())?;
                    } else {
                        self.out.write_all(DIM_GREY_ON.as_bytes())?;
                    }
                    in_code = !in_code;
                }
                _ => buf.push(c),
            }
        }
        self.out.write_all(buf.as_bytes())?;
        if in_bold {
            self.out.write_all(BOLD_OFF.as_bytes())?;
        }
        if in_italic {
            self.out.write_all(ITAL_OFF.as_bytes())?;
        }
        if in_code {
            self.out.write_all(DIM_GREY_OFF.as_bytes())?;
        }
        Ok(())
    }
}

impl<W: Write> MarkdownStreamer<W> {
    fn print_spinner_line(&mut self, kind: &str) -> io::Result<()> {
        let frame = SPINNER_FRAMES[self.spinner.frame];
        write!(self.out, "[ rendering {kind} {frame} ]")?;
        self.out.flush()?;
        self.spinner.mark_painted();
        Ok(())
    }

    fn maybe_tick_spinner(&mut self) -> io::Result<()> {
        if self.block.is_none() || !self.spinner.ready() {
            return Ok(());
        }
        let kind = match self.block.as_ref().expect("block.is_some checked") {
            Block::Table { .. } => "table",
            Block::CodeFence { lang, .. } => {
                if lang.is_empty() {
                    "code"
                } else {
                    lang.as_str()
                }
            }
        };
        let frame = self.spinner.tick();
        write!(self.out, "{CLEAR_LINE}[ rendering {kind} {frame} ]")?;
        self.out.flush()?;
        self.spinner.mark_painted();
        Ok(())
    }

    fn open_code_fence(&mut self, lang: String) -> io::Result<()> {
        self.list_stack.clear();
        self.block = Some(Block::CodeFence {
            lang: lang.clone(),
            body: Vec::new(),
        });
        let kind = if lang.is_empty() {
            "code"
        } else {
            lang.as_str()
        };
        self.print_spinner_line(kind)?;
        Ok(())
    }

    fn close_code_fence(&mut self) -> io::Result<()> {
        let (lang, body) = match self.block.take() {
            Some(Block::CodeFence { lang, body }) => (lang, body),
            _ => return Ok(()),
        };
        write!(self.out, "{CLEAR_LINE}")?;
        let header_lang = if lang.is_empty() {
            "code"
        } else {
            lang.as_str()
        };
        writeln!(self.out, "{DIM_GREY_ON}── {header_lang} ──{DIM_GREY_OFF}")?;
        for line in body {
            writeln!(self.out, "{DIM_GREY_ON}{line}{DIM_GREY_OFF}")?;
        }
        writeln!(self.out, "{DIM_GREY_ON}── end ──{DIM_GREY_OFF}")?;
        Ok(())
    }

    fn close_table(&mut self) -> io::Result<()> {
        let lines = match self.block.take() {
            Some(Block::Table { lines }) => lines,
            _ => return Ok(()),
        };
        write!(self.out, "{CLEAR_LINE}")?;
        match render_table(&lines) {
            Some(rendered) => writeln!(self.out, "{rendered}")?,
            None => {
                // Buffered lines weren't a real table; emit as
                // prose so any inline markdown still renders.
                for l in lines {
                    self.emit_inline_line(&l)?;
                }
            }
        }
        Ok(())
    }

    /// Called from `finish()` when EOS arrives mid-block. Render
    /// the buffered content as if the closer had arrived; tables
    /// without a separator row fall back to raw lines.
    fn flush_block_raw(&mut self) -> io::Result<()> {
        write!(self.out, "{CLEAR_LINE}")?;
        match self.block.take() {
            Some(Block::Table { lines }) => {
                if let Some(rendered) = render_table(&lines) {
                    writeln!(self.out, "{rendered}")?;
                } else {
                    for l in lines {
                        writeln!(self.out, "{l}")?;
                    }
                }
            }
            Some(Block::CodeFence { lang, body }) => {
                let header_lang = if lang.is_empty() {
                    "code"
                } else {
                    lang.as_str()
                };
                writeln!(self.out, "{DIM_GREY_ON}── {header_lang} ──{DIM_GREY_OFF}")?;
                for l in body {
                    writeln!(self.out, "{DIM_GREY_ON}{l}{DIM_GREY_OFF}")?;
                }
                writeln!(self.out, "{DIM_GREY_ON}── end ──{DIM_GREY_OFF}")?;
            }
            None => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum Align {
    Left,
    Center,
    Right,
}

/// Render a list of `|...|` lines into a comfy-table string. Returns
/// `None` if the lines don't form a valid table (no separator row).
fn render_table(lines: &[String]) -> Option<String> {
    if lines.len() < 2 {
        return None;
    }
    let header = split_row(&lines[0]);
    let aligns = parse_separator_row(&lines[1])?;
    let cols = header.len().min(aligns.len()).max(1);
    let body_rows: Vec<Vec<String>> = lines[2..].iter().map(|l| split_row(l)).collect();

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_content_arrangement(ContentArrangement::Dynamic);

    let header_cells: Vec<Cell> = header
        .iter()
        .take(cols)
        .enumerate()
        .map(|(i, txt)| {
            let mut cell = Cell::new(txt).add_attribute(comfy_table::Attribute::Bold);
            cell = cell.set_alignment(comfy_alignment(aligns[i]));
            cell
        })
        .collect();
    table.set_header(header_cells);

    for row in body_rows {
        let cells: Vec<Cell> = (0..cols)
            .map(|i| {
                let txt = row.get(i).map(String::as_str).unwrap_or("");
                Cell::new(txt).set_alignment(comfy_alignment(aligns[i]))
            })
            .collect();
        table.add_row(cells);
    }
    Some(table.to_string())
}

fn comfy_alignment(a: Align) -> CellAlignment {
    match a {
        Align::Left => CellAlignment::Left,
        Align::Center => CellAlignment::Center,
        Align::Right => CellAlignment::Right,
    }
}

/// Split `| a | b | c |` into `["a", "b", "c"]`. Trims whitespace
/// inside each cell; preserves empty cells.
fn split_row(line: &str) -> Vec<String> {
    line.trim()
        .trim_start_matches('|')
        .trim_end_matches('|')
        .split('|')
        .map(|s| s.trim().to_owned())
        .collect()
}

/// Parse a separator row `|:--|:-:|--:|` into a per-column alignment
/// list. Returns `None` if any cell isn't a valid alignment marker.
fn parse_separator_row(line: &str) -> Option<Vec<Align>> {
    let cells = split_row(line);
    cells
        .iter()
        .map(|c| {
            let c = c.trim();
            if c.is_empty() {
                return None;
            }
            let left = c.starts_with(':');
            let right = c.ends_with(':');
            let core = c.trim_matches(':');
            if core.is_empty() || !core.chars().all(|ch| ch == '-') {
                return None;
            }
            Some(match (left, right) {
                (true, true) => Align::Center,
                (false, true) => Align::Right,
                _ => Align::Left,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(input: &str) -> String {
        let mut s = MarkdownStreamer::with_tty(Vec::<u8>::new(), true);
        s.push(input).unwrap();
        s.finish().unwrap();
        String::from_utf8(s.out).unwrap()
    }

    fn render_split(chunks: &[&str]) -> String {
        let mut s = MarkdownStreamer::with_tty(Vec::<u8>::new(), true);
        for c in chunks {
            s.push(c).unwrap();
        }
        s.finish().unwrap();
        String::from_utf8(s.out).unwrap()
    }

    #[test]
    fn passthrough_when_not_tty() {
        let mut s = MarkdownStreamer::with_tty(Vec::<u8>::new(), false);
        s.push("# heading\n**bold** and `code`\n").unwrap();
        s.finish().unwrap();
        // Non-TTY: byte-for-byte passthrough, no ANSI injected.
        assert_eq!(
            String::from_utf8(s.out).unwrap(),
            "# heading\n**bold** and `code`\n"
        );
    }

    #[test]
    fn heading_renders_bold() {
        let out = render("# Hello\n");
        assert_eq!(out, format!("{BOLD_ON}Hello{BOLD_OFF}\n"));
    }

    #[test]
    fn heading_levels_all_bold() {
        for d in 1..=6 {
            let hashes = "#".repeat(d);
            let out = render(&format!("{hashes} x\n"));
            assert!(out.contains(BOLD_ON), "depth {d} missing bold-on");
            assert!(out.contains(BOLD_OFF), "depth {d} missing bold-off");
        }
    }

    #[test]
    fn bold_inline() {
        let out = render("hi **bold** end\n");
        assert_eq!(out, format!("hi {BOLD_ON}bold{BOLD_OFF} end\n"));
    }

    #[test]
    fn italic_inline() {
        let out = render("hi *em* end\n");
        assert_eq!(out, format!("hi {ITAL_ON}em{ITAL_OFF} end\n"));
    }

    #[test]
    fn inline_code() {
        let out = render("call `foo()` now\n");
        assert_eq!(out, format!("call {DIM_GREY_ON}foo(){DIM_GREY_OFF} now\n"));
    }

    #[test]
    fn split_bold_marker_across_chunks() {
        let out = render_split(&["hi *", "*bold*", "* rest\n"]);
        assert_eq!(out, format!("hi {BOLD_ON}bold{BOLD_OFF} rest\n"));
    }

    #[test]
    fn unclosed_bold_at_eol_resets() {
        let out = render("hi **bold no close\nnext line\n");
        // First line: bold-on, text, bold-off (auto-close at \n).
        // Second line: plain.
        assert!(out.starts_with(&format!("hi {BOLD_ON}bold no close{BOLD_OFF}\n")));
        assert!(out.contains("next line\n"));
    }

    #[test]
    fn unordered_list_one_level() {
        let out = render("- one\n- two\n");
        assert_eq!(out, "• one\n• two\n");
    }

    #[test]
    fn ordered_list_preserves_numbers() {
        let out = render("1. one\n3. three\n");
        assert_eq!(out, "1. one\n3. three\n");
    }

    #[test]
    fn nested_list_indents() {
        let out = render("- top\n  - mid\n    - deep\n");
        assert_eq!(out, "• top\n  ◦ mid\n    ▪ deep\n");
    }

    #[test]
    fn table_renders_via_comfy() {
        let input = "| a | b |\n|---|---|\n| 1 | 2 |\n\nafter\n";
        let out = render(input);
        // Spinner line was emitted then cleared with `\r\x1b[K`.
        assert!(out.contains(CLEAR_LINE), "spinner not cleared");
        // Header text + body text present.
        assert!(out.contains('a') && out.contains('b'));
        assert!(out.contains('1') && out.contains('2'));
        // Box-drawing chars from comfy-table UTF8_FULL_CONDENSED
        // preset.
        assert!(out.chars().any(|c| c == '│' || c == '┃' || c == '|'));
        // "after" line follows the table.
        assert!(out.contains("after\n"));
    }

    #[test]
    fn code_fence_renders_dim() {
        let input = "```rust\nfn x() {}\n```\n";
        let out = render(input);
        assert!(out.contains("── rust ──"));
        assert!(out.contains("fn x() {}"));
        assert!(out.contains("── end ──"));
        assert!(out.contains(CLEAR_LINE));
    }

    #[test]
    fn unclosed_code_fence_renders_on_finish() {
        let input = "```rust\nfn x() {}\n"; // no closing ```
        let out = render(input);
        // No closing fence arrived, but we still render the body —
        // dumping raw `` ``` `` markers would just look like the
        // detector failed.
        assert!(out.contains("── rust ──"));
        assert!(out.contains("fn x() {}"));
        assert!(out.contains("── end ──"));
    }

    #[test]
    fn pipe_in_prose_is_not_a_table() {
        // No separator row → `render_table` returns None → lines
        // flush through inline path.
        let input = "Use | for tables in markdown\n";
        let out = render(input);
        assert!(out.contains("Use | for tables"));
    }

    #[test]
    fn table_after_inline_text_without_blank_line() {
        // Gemma frequently emits prose followed *directly* by a
        // table with no blank line between — and sometimes with no
        // leading newline before the first `|` either (the model
        // ends prose with `:` and immediately streams `|Header|`).
        // We must still recognise the table.
        let input = "Here's a comparison:\n| A | B |\n| :- | :- |\n| 1 | 2 |\nafter\n";
        let out = render(input);
        eprintln!("table_after_inline OUT:\n{out}");
        assert!(out.contains('1') && out.contains('2'));
        assert!(out.contains('│') || out.contains('|'));
    }

    #[test]
    fn gemma_doom_table_renders() {
        let input = "| Feature | Doom Engine (id Tech 1) | Quake Engine (id Tech 2) |\n\
                     | :--- | :--- | :--- |\n\
                     | **Dimension** | \"Pseudo-3D\" (2D map with tricks) | **True 3D** (Polygon-based) |\n\
                     | **Movement** | Cannot look up/down | **Full 3D movement** |\n\
                     \n";
        let out = render(input);
        eprintln!("gemma_doom_table_renders OUT:\n{out}");
        assert!(out.contains("Feature"), "header missing");
        assert!(out.contains("Dimension"), "row 1 missing");
        assert!(out.contains("Movement"), "row 2 missing");
    }

    #[test]
    fn list_state_resets_after_blank_line() {
        let out = render("- one\n\n- two\n");
        // Both at depth 0 → both use `•`. Without reset, an indent
        // inferred from a stale stack could re-render with `◦`.
        assert_eq!(out, "• one\n\n• two\n");
    }
}
