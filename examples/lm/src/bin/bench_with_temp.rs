//! Wrap a `cargo bench` invocation with `macmon raw` sampling, then
//! emit a CSV of `(t_seconds, gpu_C, cpu_C, gpu_power_W, sys_power_W)`
//! and a PNG plot overlaying temps + power vs time, with bench-cell
//! boundaries annotated.
//!
//! Usage:
//!   bench_with_temp --bench-args '-- ^gemma4_decode_26b_a4b_it_q8/' \
//!                   --interval-ms 1000 --out /tmp/gemma_temp
//!
//! Output: /tmp/gemma_temp.csv, /tmp/gemma_temp.png, /tmp/gemma_temp.bench.log

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use plotters::prelude::*;
use serde::Deserialize;

type BoxErr = Box<dyn std::error::Error + Send + Sync>;
type Res<T> = std::result::Result<T, BoxErr>;

#[derive(Debug, Deserialize)]
struct MacmonSample {
    temp: Temp,
    gpu_power: f64,
    sys_power: f64,
    memory: Memory,
}

#[derive(Debug, Deserialize)]
struct Temp {
    cpu_temp_avg: f64,
    gpu_temp_avg: f64,
}

#[derive(Debug, Deserialize)]
struct Memory {
    ram_usage: u64,
    swap_usage: u64,
}

#[derive(Debug, Clone)]
struct Reading {
    t_s: f64,
    gpu_c: f64,
    cpu_c: f64,
    gpu_w: f64,
    sys_w: f64,
    ram_gb: f64,
    swap_gb: f64,
}

#[derive(Debug, Clone)]
struct BenchEvent {
    t_s: f64,
    label: String,
}

/// `[mlx_mem] <tag> active_mb=<X> cache_mb=<Y> peak_mb=<Z>` emitted by
/// the bench harness at cell start/end. Charted as a separate series
/// on top of macmon's system-wide RAM.
#[derive(Debug, Clone)]
struct MlxMemEvent {
    t_s: f64,
    tag: String,
    active_mb: f64,
    cache_mb: f64,
    peak_mb: f64,
}

struct Args {
    bench_args: String,
    interval_ms: u64,
    out_prefix: PathBuf,
}

/// Resolve the cargo workspace root at compile time. `CARGO_MANIFEST_DIR`
/// is `examples/lm`; the workspace root is two levels up.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn parse_args() -> Res<Args> {
    let mut bench_args = String::new();
    let mut interval_ms: u64 = 1000;
    let mut out_prefix = PathBuf::from("/tmp/bench_temp");
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bench-args" => bench_args = it.next().ok_or("--bench-args needs a value")?,
            "--interval-ms" => {
                interval_ms = it.next().ok_or("--interval-ms needs a value")?.parse()?
            }
            "--out" => out_prefix = PathBuf::from(it.next().ok_or("--out needs a value")?),
            "-h" | "--help" => {
                eprintln!(
                    "bench_with_temp --bench-args '<cargo-bench args>' \\\n  [--interval-ms 1000] [--out /tmp/bench_temp]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }
    if bench_args.is_empty() {
        return Err("--bench-args is required".into());
    }
    Ok(Args {
        bench_args,
        interval_ms,
        out_prefix,
    })
}

fn spawn_macmon(
    interval_ms: u64,
    tx: mpsc::Sender<Reading>,
    start: Instant,
) -> Res<std::process::Child> {
    let mut child = Command::new("macmon")
        .args(["raw", "--interval", &interval_ms.to_string()])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let stdout = child.stdout.take().ok_or("macmon stdout missing")?;
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(|l| l.ok()) {
            if line.is_empty() {
                continue;
            }
            let parsed: serde_json::Result<MacmonSample> = serde_json::from_str(&line);
            let Ok(sample) = parsed else { continue };
            let r = Reading {
                t_s: start.elapsed().as_secs_f64(),
                gpu_c: sample.temp.gpu_temp_avg,
                cpu_c: sample.temp.cpu_temp_avg,
                gpu_w: sample.gpu_power,
                sys_w: sample.sys_power,
                ram_gb: sample.memory.ram_usage as f64 / 1e9,
                swap_gb: sample.memory.swap_usage as f64 / 1e9,
            };
            if tx.send(r).is_err() {
                break;
            }
        }
    });
    Ok(child)
}

fn parse_mlx_mem(line: &str, t_s: f64) -> Option<MlxMemEvent> {
    // Expected: "[mlx_mem] <tag> active_mb=<x> cache_mb=<y> peak_mb=<z>"
    let rest = line.strip_prefix("[mlx_mem] ")?;
    let mut parts = rest.splitn(2, ' ');
    let tag = parts.next()?.to_string();
    let kv = parts.next()?;
    let mut active_mb = None;
    let mut cache_mb = None;
    let mut peak_mb = None;
    for token in kv.split_whitespace() {
        let (k, v) = token.split_once('=')?;
        let val: f64 = v.parse().ok()?;
        match k {
            "active_mb" => active_mb = Some(val),
            "cache_mb" => cache_mb = Some(val),
            "peak_mb" => peak_mb = Some(val),
            _ => {}
        }
    }
    Some(MlxMemEvent {
        t_s,
        tag,
        active_mb: active_mb?,
        cache_mb: cache_mb?,
        peak_mb: peak_mb?,
    })
}

fn run_bench(
    bench_args: &str,
    log_path: &PathBuf,
    start: Instant,
) -> Res<(Vec<BenchEvent>, Vec<MlxMemEvent>)> {
    // criterion writes "Benchmarking ..." to stderr; pipe both streams
    // and tee them line-by-line to the log file while extracting events.
    let mut child = Command::new("cargo")
        .arg("bench")
        .arg("-p")
        .arg("mlx-lm")
        .arg("--bench")
        .arg("lm_decode")
        .args(bench_args.split_whitespace())
        .env("MLX_LM_BENCH_NO_DOWNLOAD", "1")
        .env("MLX_LM_BENCH_SET", "trimmed")
        .current_dir(workspace_root())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = child.stdout.take().ok_or("cargo bench stdout missing")?;
    let stderr = child.stderr.take().ok_or("cargo bench stderr missing")?;

    let (ev_tx, ev_rx) = mpsc::channel::<BenchEvent>();
    let (mem_tx, mem_rx) = mpsc::channel::<MlxMemEvent>();
    let log_handle = File::create(log_path)?;
    let log_arc = std::sync::Arc::new(std::sync::Mutex::new(log_handle));

    let spawn_reader = |stream: Box<dyn std::io::Read + Send>,
                        tag: &'static str,
                        ev: mpsc::Sender<BenchEvent>,
                        mem: mpsc::Sender<MlxMemEvent>,
                        log: std::sync::Arc<std::sync::Mutex<File>>,
                        start: Instant|
     -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let reader = BufReader::new(stream);
            for line in reader.lines().map_while(|l| l.ok()) {
                if let Ok(mut f) = log.lock() {
                    let _ = writeln!(f, "[{tag}] {line}");
                }
                let t_s = start.elapsed().as_secs_f64();
                if let Some(stripped) = line.strip_prefix("Benchmarking ") {
                    if !stripped.contains(": Warming")
                        && !stripped.contains(": Collecting")
                        && !stripped.contains(": Analyzing")
                    {
                        let _ = ev.send(BenchEvent {
                            t_s,
                            label: stripped.to_string(),
                        });
                    }
                } else if let Some(m) = parse_mlx_mem(&line, t_s) {
                    let _ = mem.send(m);
                }
            }
        })
    };

    let h_out = spawn_reader(
        Box::new(stdout),
        "out",
        ev_tx.clone(),
        mem_tx.clone(),
        log_arc.clone(),
        start,
    );
    let h_err = spawn_reader(Box::new(stderr), "err", ev_tx, mem_tx, log_arc, start);

    let status = child.wait()?;
    let _ = h_out.join();
    let _ = h_err.join();
    if !status.success() {
        eprintln!("warning: cargo bench exited with status {status}");
    }

    let mut events = Vec::new();
    while let Ok(ev) = ev_rx.try_recv() {
        events.push(ev);
    }
    events.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Same cell may emit multiple "Benchmarking <X>" headers (warmup +
    // collect + analyze separately) — dedupe consecutive duplicates.
    events.dedup_by(|a, b| a.label == b.label);

    let mut mem_events: Vec<MlxMemEvent> = mem_rx.try_iter().collect();
    mem_events.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok((events, mem_events))
}

fn write_csv(
    path: &PathBuf,
    readings: &[Reading],
    events: &[BenchEvent],
    mem_events: &[MlxMemEvent],
) -> Res<()> {
    let mut f = File::create(path)?;
    writeln!(f, "# bench events (t_s, label):")?;
    for ev in events {
        writeln!(f, "# {:.2}\t{}", ev.t_s, ev.label)?;
    }
    writeln!(
        f,
        "# mlx_mem stamps (t_s, tag, active_mb, cache_mb, peak_mb):"
    )?;
    for m in mem_events {
        writeln!(
            f,
            "# {:.2}\t{}\t{:.1}\t{:.1}\t{:.1}",
            m.t_s, m.tag, m.active_mb, m.cache_mb, m.peak_mb
        )?;
    }
    writeln!(f, "t_s,gpu_c,cpu_c,gpu_w,sys_w,ram_gb,swap_gb")?;
    for r in readings {
        writeln!(
            f,
            "{:.3},{:.2},{:.2},{:.3},{:.3},{:.2},{:.2}",
            r.t_s, r.gpu_c, r.cpu_c, r.gpu_w, r.sys_w, r.ram_gb, r.swap_gb
        )?;
    }
    Ok(())
}

fn render_svg(
    path: &PathBuf,
    readings: &[Reading],
    events: &[BenchEvent],
    mem_events: &[MlxMemEvent],
) -> Res<()> {
    if readings.is_empty() {
        return Err("no readings to plot".into());
    }
    let t_max = readings.last().unwrap().t_s.max(1.0);
    let temp_min = readings
        .iter()
        .map(|r| r.cpu_c.min(r.gpu_c))
        .fold(f64::INFINITY, f64::min);
    let temp_max = readings
        .iter()
        .map(|r| r.cpu_c.max(r.gpu_c))
        .fold(f64::NEG_INFINITY, f64::max);
    let pw_max = readings
        .iter()
        .map(|r| r.sys_w)
        .fold(0.0_f64, f64::max)
        .max(1.0);

    let t_lo = (temp_min - 2.0).floor();
    let t_hi = (temp_max + 2.0).ceil();

    let mlx_peak_gb = mem_events
        .iter()
        .map(|m| m.peak_mb.max(m.active_mb + m.cache_mb) / 1024.0)
        .fold(0.0_f64, f64::max);
    let mem_max = readings
        .iter()
        .map(|r| (r.ram_gb + r.swap_gb).max(r.ram_gb))
        .fold(0.0_f64, f64::max)
        .max(mlx_peak_gb)
        .max(1.0);

    let root = SVGBackend::new(path, (1600, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    let panes = root.split_evenly((3, 1));

    // ---- Pane 0: temperature vs time ----
    let mut chart = ChartBuilder::on(&panes[0])
        .caption("Temp (°C) — GPU red, CPU blue", ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..t_max, t_lo..t_hi)?;
    chart.configure_mesh().x_desc("t (s)").y_desc("°C").draw()?;
    chart.draw_series(LineSeries::new(
        readings.iter().map(|r| (r.t_s, r.gpu_c)),
        RED.stroke_width(2),
    ))?;
    chart.draw_series(LineSeries::new(
        readings.iter().map(|r| (r.t_s, r.cpu_c)),
        BLUE.stroke_width(2),
    ))?;
    for ev in events {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(ev.t_s, t_lo), (ev.t_s, t_hi)],
            BLACK.mix(0.25).stroke_width(1),
        )))?;
    }

    // ---- Pane 1: power vs time ----
    let mut chart2 = ChartBuilder::on(&panes[1])
        .caption("Power (W) — GPU magenta, sys green", ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..t_max, 0f64..pw_max)?;
    chart2.configure_mesh().x_desc("t (s)").y_desc("W").draw()?;
    chart2.draw_series(LineSeries::new(
        readings.iter().map(|r| (r.t_s, r.gpu_w)),
        MAGENTA.stroke_width(2),
    ))?;
    chart2.draw_series(LineSeries::new(
        readings.iter().map(|r| (r.t_s, r.sys_w)),
        GREEN.stroke_width(2),
    ))?;
    for ev in events {
        chart2.draw_series(std::iter::once(PathElement::new(
            vec![(ev.t_s, 0.0), (ev.t_s, pw_max)],
            BLACK.mix(0.25).stroke_width(1),
        )))?;
    }

    // ---- Pane 2: memory vs time ----
    let mut chart3 = ChartBuilder::on(&panes[2])
        .caption(
            "Memory (GB) — sys-RAM cyan, swap orange, MLX-active purple, MLX-cache grey",
            ("sans-serif", 18),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..t_max, 0f64..mem_max)?;
    chart3
        .configure_mesh()
        .x_desc("t (s)")
        .y_desc("GB")
        .draw()?;
    chart3.draw_series(LineSeries::new(
        readings.iter().map(|r| (r.t_s, r.ram_gb)),
        CYAN.stroke_width(2),
    ))?;
    chart3.draw_series(LineSeries::new(
        readings.iter().map(|r| (r.t_s, r.swap_gb)),
        RGBColor(255, 140, 0).stroke_width(2),
    ))?;
    if !mem_events.is_empty() {
        chart3.draw_series(LineSeries::new(
            mem_events.iter().map(|m| (m.t_s, m.active_mb / 1024.0)),
            RGBColor(140, 80, 200).stroke_width(2),
        ))?;
        chart3.draw_series(LineSeries::new(
            mem_events
                .iter()
                .map(|m| (m.t_s, (m.active_mb + m.cache_mb) / 1024.0)),
            RGBColor(110, 110, 110).stroke_width(1),
        ))?;
        chart3.draw_series(mem_events.iter().map(|m| {
            Circle::new(
                (m.t_s, m.active_mb / 1024.0),
                3,
                RGBColor(140, 80, 200).filled(),
            )
        }))?;
    }
    for ev in events {
        chart3.draw_series(std::iter::once(PathElement::new(
            vec![(ev.t_s, 0.0), (ev.t_s, mem_max)],
            BLACK.mix(0.25).stroke_width(1),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn main() -> Res<()> {
    let args = parse_args()?;
    let csv_path = args.out_prefix.with_extension("csv");
    let svg_path = args.out_prefix.with_extension("svg");
    let log_path = args.out_prefix.with_extension("bench.log");

    let start = Instant::now();
    let (tx, rx) = mpsc::channel::<Reading>();
    let mut macmon = spawn_macmon(args.interval_ms, tx, start)?;

    eprintln!(
        "[bench_with_temp] sampling every {} ms — running bench…",
        args.interval_ms
    );
    let (events, mem_events) = run_bench(&args.bench_args, &log_path, start)?;

    // Give macmon a final tick then kill it.
    thread::sleep(Duration::from_millis(args.interval_ms + 200));
    let _ = macmon.kill();
    let _ = macmon.wait();

    let mut readings: Vec<Reading> = rx.try_iter().collect();
    readings.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    eprintln!(
        "[bench_with_temp] {} readings, {} bench events, {} mlx_mem stamps, {:.1}s elapsed",
        readings.len(),
        events.len(),
        mem_events.len(),
        start.elapsed().as_secs_f64()
    );

    write_csv(&csv_path, &readings, &events, &mem_events)?;
    render_svg(&svg_path, &readings, &events, &mem_events)?;

    if !readings.is_empty() {
        let stats = |f: &dyn Fn(&Reading) -> f64| -> (f64, f64, f64) {
            let vs: Vec<f64> = readings.iter().map(f).collect();
            let mn = vs.iter().copied().fold(f64::INFINITY, f64::min);
            let mx = vs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mean = vs.iter().sum::<f64>() / vs.len() as f64;
            (mn, mean, mx)
        };
        let (gmin, gmean, gmax) = stats(&|r: &Reading| r.gpu_c);
        let (_, pmean, pmax) = stats(&|r: &Reading| r.gpu_w);
        let (rmin, rmean, rmax) = stats(&|r: &Reading| r.ram_gb);
        let (smin, _, smax) = stats(&|r: &Reading| r.swap_gb);
        eprintln!(
            "[bench_with_temp] GPU temp: min {gmin:.1}°C  mean {gmean:.1}°C  max {gmax:.1}°C"
        );
        eprintln!("[bench_with_temp] GPU power: mean {pmean:.1} W  peak {pmax:.1} W");
        eprintln!(
            "[bench_with_temp] RAM: {rmin:.1}–{rmax:.1} GB (mean {rmean:.1})  |  swap: {smin:.1}–{smax:.1} GB"
        );
        if !mem_events.is_empty() {
            let mlx_active_max = mem_events
                .iter()
                .map(|m| m.active_mb)
                .fold(0.0_f64, f64::max);
            let mlx_active_min = mem_events
                .iter()
                .map(|m| m.active_mb)
                .fold(f64::INFINITY, f64::min);
            let mlx_cache_max = mem_events
                .iter()
                .map(|m| m.cache_mb)
                .fold(0.0_f64, f64::max);
            let mlx_peak_max = mem_events.iter().map(|m| m.peak_mb).fold(0.0_f64, f64::max);
            eprintln!(
                "[bench_with_temp] MLX active: {:.2}–{:.2} GB  cache peak: {:.2} GB  reported peak: {:.2} GB",
                mlx_active_min / 1024.0,
                mlx_active_max / 1024.0,
                mlx_cache_max / 1024.0,
                mlx_peak_max / 1024.0,
            );
            eprintln!(
                "[bench_with_temp] gap MLX→sys-RAM: {:.1} GB at sys-peak (driver/IOAccel overhead)",
                rmax - (mlx_active_max + mlx_cache_max) / 1024.0,
            );
        }
    }

    eprintln!("[bench_with_temp] csv: {}", csv_path.display());
    eprintln!("[bench_with_temp] svg: {}", svg_path.display());
    eprintln!("[bench_with_temp] bench log: {}", log_path.display());
    Ok(())
}
