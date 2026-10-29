//! inferno-bench: scheduler-layer benchmarks.
//!
//! Two modes:
//!   inferno-bench load        — burst load through the continuous batcher
//!                               across concurrency levels; reports
//!                               throughput, TTFT/TPOT percentiles,
//!                               preemptions, prefix-reuse rate.
//!   inferno-bench threads     — verifies the worker pool scales (map-reduce
//!                               sum-of-squares across 1..=N threads).
//!
//! Scheduling-relevant cost only: the pseudo-model reproduces prefill-~
//! linear / decode-constant cost shapes and real K/V write traffic. It is
//! NOT a language model — model-quality work comes with the real backend.
//! Runs are deterministic (seeded xorshift).

use std::io::Write;
use std::sync::mpsc;
use std::time::Instant;

use inferno_core::batching::{Batcher, BatcherConfig, Priority, Submit};
use inferno_core::kernels::matmul_blocked;
use inferno_core::kv_cache::KvConfig;
use inferno_core::pseudo::PseudoEngine;
use inferno_core::sched::{Admission, AdmissionClass, NullAdmission};
use inferno_core::threads::ThreadPool;
use inferno_model::QwenEngine;

#[cfg(target_os = "linux")]
use inferno_core::sched_linux::ForesightAdmission;

fn args() -> Vec<String> {
    std::env::args().skip(1).collect()
}

fn flag(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state | 1;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() as f64) * p).clamp(0.0, v.len() as f64 - 1.0) as usize;
    v[idx]
}

fn build_admission() -> (std::sync::Arc<dyn Admission>, &'static str) {
    #[cfg(target_os = "linux")]
    {
        if let Some(a) = ForesightAdmission::detect() {
            let live = a.is_live();
            return (
                std::sync::Arc::new(a),
                if live { "foresight-live" } else { "foresight-detected-dead" },
            );
        }
        return (
            std::sync::Arc::new(NullAdmission),
            "off (no /sys/fs/bpf/foresight)",
        );
    }
    #[cfg(not(target_os = "linux"))]
    {
        (std::sync::Arc::new(NullAdmission), "off (non-linux host)")
    }
}

fn cmd_threads() {
    let threads: usize = flag(&args(), "--threads", "8").parse().unwrap_or(8);
    let n: usize = 4_000_000;
    let items: Vec<u64> = (0..n as u64).collect();

    let t0 = Instant::now();
    let single: u64 = items.iter().map(|&x| x.wrapping_mul(x)).sum();
    let t_single = t0.elapsed();

    let pool = ThreadPool::new(Some(threads)).unwrap();
    let t0 = Instant::now();
    let partial: u64 = pool
        .map_chunks(items, |chunk, _id| chunk.iter().map(|&x| x.wrapping_mul(x)).sum::<u64>())
        .into_iter()
        .sum();
    let t_pool = t0.elapsed();
    let got = partial;

    println!("threads mode");
    println!("  elements            : {}", n);
    println!("  single-thread       : {:>8.1} ms  sum={}", t_single.as_secs_f64() * 1e3, single);
    println!(
        "  pool ({} workers)   : {:>8.1} ms  sum={}  speedup={:.2}x",
        threads,
        t_pool.as_secs_f64() * 1e3,
        got,
        t_single.as_secs_f64() / t_pool.as_secs_f64()
    );
    assert_eq!(single, got, "pool must compute the identical sum");

    // Also verify a small matmul correctness path (kernel sanity).
    let a = vec![1.0f32; 64 * 64];
    let b = vec![1.0f32; 64 * 64];
    let c = matmul_blocked(&a, &b, 64, 64, 64, 16);
    assert_eq!(c[0], 64.0);
    println!("  matmul sanity       : ok (64x64x64 blocked)");
}

struct LoadCfg {
    requests: usize,
    clients: Vec<usize>,
    prompt_min: usize,
    prompt_max: usize,
    max_gen: usize,
    chunk_tokens: usize,
    max_num_seqs: usize,
    tag_interactive: bool,
}

fn cmd_load(admission: std::sync::Arc<dyn Admission>, sched_label: &str) {
    let a = args();
    let cfg = LoadCfg {
        requests: flag(&a, "--requests", "256").parse().unwrap_or(256),
        clients: [1usize, 4, 16, 64].to_vec(),
        prompt_min: flag(&a, "--prompt-min", "32").parse().unwrap_or(32),
        prompt_max: flag(&a, "--prompt-max", "512").parse().unwrap_or(512),
        max_gen: flag(&a, "--max-gen", "64").parse().unwrap_or(64),
        chunk_tokens: flag(&a, "--chunk", "512").parse().unwrap_or(512),
        max_num_seqs: flag(&a, "--seqs", "64").parse().unwrap_or(64),
        tag_interactive: a.iter().any(|x| x == "--tag-interactive"),
    };

    println!("load mode");
    println!("  foresight-sched     : {}", sched_label);
    if let Some(s) = admission.snapshot() {
        println!(
            "  knobs               : islice={}ns bslice={}ns duty={}pct streak={}",
            s.knobs.interactive_slice_ns,
            s.knobs.batch_slice_ns,
            s.knobs.batch_duty_pct,
            s.knobs.demote_streak
        );
        println!("  stats (n/i/b)       : {:?}", s.stats);
        if !s.demotions.is_empty() {
            println!("  demotions           : {} visible", s.demotions.len());
        }
    }

    let bs = 16usize;
    let pool = ThreadPool::new(Some(cfg.clients.iter().max().copied().unwrap_or(1))).unwrap();

    // Tag worker threads into a foresight class when live (and asked).
    // Tagging must happen INSIDE the worker (kernel maps are keyed by tid).
    if admission.is_live() && cfg.tag_interactive && std::cfg!(target_os = "linux") {
        let mut tags = 0;
        for _ in 0..pool.workers() {
            let adm = std::sync::Arc::clone(&admission);
            let ok = pool
                .spawn(move || adm.tag_thread(inferno_core::sched::current_tid(), AdmissionClass::Interactive))
                .join()
                .unwrap_or(false);
            if ok {
                tags += 1;
            }
        }
        println!("  worker tagging      : interactive ({} workers)", tags);
    }

    println!(
        "  config              : R={} gen={} prompt={}[{}..{}] chunk={} seqs={}",
        cfg.requests, cfg.max_gen, cfg.prompt_min, cfg.prompt_max, cfg.prompt_max, cfg.chunk_tokens, cfg.max_num_seqs
    );
    println!();
    println!(
        "  {:>5} | {:>10} | {:>10} | {:>9} | {:>9} | {:>7} | {:>6} | {:>5}",
        "conc", "tok/s", "req/s", "ttft_p50", "ttft_p99", "tpot_p50", "preemp", "reuse%"
    );
    println!("  ---------------------------------------------------------------");

    let mut results = Vec::new();
    for &c in &cfg.clients {
        let (tx, rx) = mpsc::channel::<Submit>();
        let seed_base = 0xfeed_c0de_cafe_0001 ^ (c as u64 * 0x9E3779B97F4A7C15);
        let r_req = cfg.requests;
        let (pmin, pmax, mgen) = (cfg.prompt_min, cfg.prompt_max, cfg.max_gen);
        let client_count = c.min(cfg.requests).max(1);
        for k in 0..client_count {
            let tx = tx.clone();
            // DISTINCT per-client seed (regression: all clients previously
            // rebound the same base and submitted identical streams — the
            // reuse metric caught it).
            let mut client_seed = seed_base ^ (k as u64).wrapping_mul(0xA076_1D64_78BD_642F);
            let _ = xorshift(&mut client_seed);
            pool.spawn(move || {
                let per = r_req / client_count;
                for r in 0..per {
                    let mut s = client_seed.wrapping_add(r as u64 * 0xDA3E39CBFD3B0F1D);
                    let len = pmin + (xorshift(&mut s) as usize) % (pmax - pmin + 1).max(1);
                    let prompt: Vec<u32> = (0..len)
                        .map(|_i| (xorshift(&mut s) % 32000) as u32)
                        .collect();
                    let _ = tx.send(Submit {
                        prompt,
                        max_gen: mgen,
                        priority: Priority::Normal,
                        ttft_slo_ms: 200.0,
                    });
                }
            });
        }
        drop(tx); // channel closes when all clients finish

        // KV pool sized so 64 concurrent 512-token prompts + gen fit:
        // blocks ~= (512/16 + 64/16 + 1) * 64 => ~2300; give 4096.
        let mut batcher = Batcher::new(
            KvConfig {
                block_size: bs,
                num_blocks: 4096,
                num_layers: 4,
                num_kv_heads: 2,
                head_dim: 32,
            },
            BatcherConfig {
                prefill_chunk_tokens: cfg.chunk_tokens,
                max_num_seqs: cfg.max_num_seqs,
            },
        );
        let mut engine = PseudoEngine::new(4, 64);

        let t0 = Instant::now();
        let mut total_decode = 0usize;
        let mut total_prefill = 0usize;
        let expected = client_count * (r_req / client_count);
        let mut received = 0usize;
        loop {
            // Pump any completed client submissions in without blocking.
            while let Ok(sub) = rx.try_recv() {
                batcher.submit(sub);
                received += 1;
            }
            if received >= expected && batcher.queue_depth() == 0 {
                break;
            }
            if batcher.queue_depth() > 0 {
                let r = batcher.step(&mut engine, admission.as_ref());
                total_decode += r.decode_tokens;
                total_prefill += r.prefill_tokens;
            } else {
                // No work yet: clients haven't been scheduled. Don't spin hot.
                std::thread::yield_now();
            }
        }
        batcher.drain(&mut engine, admission.as_ref());
        let wall = t0.elapsed();

        let mut ttfts: Vec<f64> = batcher.metrics().iter().map(|m| m.ttft_us / 1e3).collect();
        let mut tpots: Vec<f64> = batcher
            .metrics()
            .iter()
            .filter(|m| m.gen_len > 1)
            .map(|m| m.tpot_us / 1e3)
            .collect();
        let reuse: f64 = batcher
            .metrics()
            .iter()
            .map(|m| m.reused_blocks as f64)
            .sum::<f64>();
        let total_blocks: f64 = batcher
            .metrics()
            .iter()
            .map(|m| m.prompt_len as f64 / bs as f64)
            .sum::<f64>()
            .max(1.0);
        let reuse_pct = 100.0 * reuse / total_blocks;

        let tok_s = total_decode as f64 / wall.as_secs_f64();
        let req_s = batcher.metrics().len() as f64 / wall.as_secs_f64();
        println!(
            "  {:>5} | {:>10.1} | {:>10.1} | {:>7.1}ms | {:>7.1}ms | {:>5.1}ms | {:>6} | {:>4.1}",
            c,
            tok_s,
            req_s,
            percentile(&mut ttfts, 0.5),
            percentile(&mut ttfts, 0.99),
            percentile(&mut tpots, 0.5),
            batcher.total_preemptions(),
            reuse_pct
        );
        results.push(serde_json_like(&batcher, c, tok_s, req_s, total_prefill, total_decode, wall.as_secs_f64(), reuse_pct));
    }

    // results.json (hand-rolled; no serde dependency)
    let json = format!(
        "[{}]",
        results.join(",\n")
    );
    let _ = std::fs::write("results.json", json);
    println!();
    println!("wrote results.json ({} regimes)", results.len());
}

fn serde_json_like(
    b: &Batcher,
    conc: usize,
    tok_s: f64,
    req_s: f64,
    prefill: usize,
    decode: usize,
    wall_s: f64,
    reuse_pct: f64,
) -> String {
    let mut ttfts: Vec<f64> = b.metrics().iter().map(|m| m.ttft_us / 1e3).collect();
    let mut tpots: Vec<f64> = b.metrics().iter().map(|m| m.tpot_us / 1e3).collect();
    let ttft_p50 = percentile(&mut ttfts, 0.5);
    let ttft_p99 = percentile(&mut ttfts, 0.99);
    let tpot_p50 = percentile(&mut tpots, 0.5);
    format!(
        r#"{{"concurrency":{},"tok_s":{:.2},"req_s":{:.2},"ttft_p50_ms":{:.2},"ttft_p99_ms":{:.2},"tpot_p50_ms":{:.2},"prefill_tokens":{},"decode_tokens":{},"preemptions":{},"reuse_pct":{:.2},"wall_s":{:.4}}}"#,
        conc, tok_s, req_s, ttft_p50, ttft_p99, tpot_p50, prefill, decode,
        b.total_preemptions(), reuse_pct, wall_s
    )
}

fn cmd_overhead() {
    let pool = ThreadPool::new(Some(8)).unwrap();
    // warm all workers
    for _ in 0..10 {
        pool.spawn(|| 1u64).join();
    }
    // noop spawn+join latency (workers parked -> wake path)
    let n = 200;
    let t0 = Instant::now();
    for _ in 0..n {
        pool.spawn(|| 1u64).join();
    }
    println!(
        "noop spawn+join: {:.3} ms each",
        t0.elapsed().as_secs_f64() * 1e3 / n as f64
    );
    // one heavy job through the pool vs inline (same work)
    let mk_items = || (0..4_000_000u64).collect::<Vec<u64>>();
    let items = mk_items();
    let t0 = Instant::now();
    let s1: u64 = items.iter().map(|&x| x.wrapping_mul(x)).sum();
    let t_inline = t0.elapsed();
    let items = mk_items();
    let t0 = Instant::now();
    let s2: u64 = pool
        .map_chunks(items, |c, _| c.iter().map(|&x| x.wrapping_mul(x)).sum::<u64>())
        .into_iter()
        .sum();
    let t_pool = t0.elapsed();
    assert_eq!(s1, s2);
    println!(
        "4M sq-sum inline {:.2} ms | through pool {:.2} ms (ratio {:.2}x)",
        t_inline.as_secs_f64() * 1e3,
        t_pool.as_secs_f64() * 1e3,
        t_pool.as_secs_f64() / t_inline.as_secs_f64()
    );

    // RAW std::thread comparison (no pool): isolates pool vs environment
    let items = mk_items();
    let t0 = Instant::now();
    let n_raw = 8usize;
    let chunk = items.len() / n_raw;
    let mut joins = Vec::new();
    for t in 0..n_raw {
        let piece: Vec<u64> = items[t * chunk..(t + 1) * chunk].to_vec();
        joins.push(std::thread::spawn(move || {
            piece.iter().map(|&x| x.wrapping_mul(x)).sum::<u64>()
        }));
    }
    let s3: u64 = joins.into_iter().map(|j| j.join().unwrap()).sum();
    let t_raw = t0.elapsed();
    assert_eq!(s1, s3);
    println!(
        "raw 8 std::threads: {:.2} ms (vs inline {:.2} ms, ratio {:.2}x)",
        t_raw.as_secs_f64() * 1e3,
        t_inline.as_secs_f64() * 1e3,
        t_raw.as_secs_f64() / t_inline.as_secs_f64()
    );
    println!(
        "available_parallelism as seen by process: {}",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
    );
}

/// Instrumented drain: per-step trace on stderr (unbuffered), livelock guard,
/// hard step cap. Returns nothing; the batcher accumulates metrics.
fn drain_traced(b: &mut Batcher, e: &mut QwenEngine, adm: &dyn Admission, quiet: bool) {
    let mut steps = 0usize;
    let mut idle = 0usize;
    loop {
        if b.queue_depth() == 0 {
            break;
        }
        let r = b.step(e, adm);
        steps += 1;
        if r.admitted == 0 && r.prefill_tokens == 0 && r.decode_tokens == 0 && r.preempted == 0
        {
            idle += 1;
            if idle > 100 {
                eprintln!("  [trace] LIVELOCK: 100 idle steps (total {steps})");
                break;
            }
        } else {
            idle = 0;
            if !quiet {
                eprintln!(
                    "  [trace] step {steps}: admitted {} prefill {} decode {} preempt {}",
                    r.admitted, r.prefill_tokens, r.decode_tokens, r.preempted
                );
            }
        }
        if steps >= 200_000 {
            eprintln!("  [trace] STEP CAP HIT — aborting drain");
            break;
        }
    }
    eprintln!("  [trace] drain done: {steps} steps");
}

/// Machine-state snapshot for outlier detection: a mid-benchmark VM restart
/// (uptime discontinuity) or host memory pressure shows up here, not as a
/// silent lie in the numbers.
fn machine_state() -> String {
    let up = std::fs::read_to_string("/proc/uptime").unwrap_or_default();
    let up = up.split_whitespace().next().unwrap_or("?").to_string();
    let la = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    let la = la.split_whitespace().take(3).collect::<Vec<_>>().join(" ");
    let mem = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let avail = mem
        .lines()
        .find(|l| l.starts_with("MemAvailable"))
        .and_then(|l| l.split_whitespace().nth(1).map(|s| s.to_string()))
        .unwrap_or_else(|| "?".into());
    format!("uptime_s={up} loadavg=({la}) mem_avail_kb={avail}")
}

/// Real-model benchmark: synthetic prompts (valid token ids, no semantics —
/// performance only), per-request CSV rows appended to --csv, machine state
/// logged per rep. Two submission modes:
///   default      conc requests submitted together, per-rep FRESH prompt
///                (seed advances per rep, so rep 2+ never reuses rep 1's
///                registered prefix — cold prefill every rep);
///   --stagger 1  the SAME prompt every round, submitted sequentially —
///                rounds 2+ measure warm TTFT via the retained prefix cache.
fn cmd_bench(admission: std::sync::Arc<dyn Admission>, sched_label: &str) {
    use std::io::Write;
    let a = args();
    let dir = flag(&a, "--model-dir", "/root/models/qwen2.5-0.5b");
    let prompt_len: usize = flag(&a, "--prompt-len", "32").parse().unwrap_or(32);
    let max_gen: usize = flag(&a, "--max-gen", "16").parse().unwrap_or(16);
    let conc: usize = flag(&a, "--conc", "1").parse().unwrap_or(1);
    let reps: usize = flag(&a, "--reps", "1").parse().unwrap_or(1);
    let stagger: bool = flag(&a, "--stagger", "0") == "1";
    let blocks: usize = flag(&a, "--blocks", "256").parse().unwrap_or(256);
    let chunk: usize = flag(&a, "--chunk", "64").parse().unwrap_or(64);
    let tag = flag(&a, "--tag", "w");
    let csv = flag(&a, "--csv", "");
    let seed: u64 = flag(&a, "--seed", "42").parse().unwrap_or(42);

    println!("bench mode — loading {dir} ...");
    let mut engine = QwenEngine::load(std::path::Path::new(&dir)).expect("model load failed");
    let cfg = engine.cfg.clone();
    let mut batcher = Batcher::new(
        KvConfig {
            block_size: 16,
            num_blocks: blocks,
            num_layers: cfg.num_layers,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
        },
        BatcherConfig {
            prefill_chunk_tokens: chunk,
            max_num_seqs: 8,
        },
    );
    eprintln!(
        "[bench] sched={sched_label} gemv_workers={} {}",
        engine.pool_workers(),
        machine_state()
    );

    let mut csv_file = if csv.is_empty() {
        None
    } else {
        let new = !std::path::Path::new(&csv).exists()
            || std::fs::metadata(&csv).map(|m| m.len() == 0).unwrap_or(true);
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&csv)
            .expect("csv open failed");
        if new {
            let mut f = f;
            writeln!(
                f,
                "kind,tag,rep,id,prompt_len,gen_len,ttft_ms,tpot_ms,reuse,preempt,wall_s,tokens,uptime_s"
            )
            .unwrap();
            Some(f)
        } else {
            Some(f)
        }
    };

    let mut st = seed;
    for rep in 0..reps {
        // Non-stagger: fresh prompt per rep so no rep inherits another's
        // registered prefix (cold prefill is the thing being measured).
        // Stagger: identical prompt — the reuse IS the measurement.
        let prompt: Vec<u32> = if stagger {
            let mut s2 = seed;
            (0..prompt_len)
                .map(|_| (xorshift(&mut s2) % 100_000) as u32 + 100)
                .collect()
        } else {
            (0..prompt_len)
                .map(|_| (xorshift(&mut st) % 100_000) as u32 + 100)
                .collect()
        };
        eprintln!("[bench] rep {rep} start {}", machine_state());
        let t0 = Instant::now();
        let mut ids = Vec::new();
        if stagger {
            for _ in 0..conc {
                let id = batcher.submit(Submit {
                    prompt: prompt.clone(),
                    max_gen,
                    priority: Priority::Normal,
                    ttft_slo_ms: 60_000.0,
                });
                ids.push(id);
                drain_traced(&mut batcher, &mut engine, admission.as_ref(), true);
            }
        } else {
            for _ in 0..conc {
                ids.push(batcher.submit(Submit {
                    prompt: prompt.clone(),
                    max_gen,
                    priority: Priority::Normal,
                    ttft_slo_ms: 60_000.0,
                }));
            }
            drain_traced(&mut batcher, &mut engine, admission.as_ref(), true);
        }
        let wall = t0.elapsed().as_secs_f64();
        let tokens: usize = ids
            .iter()
            .filter_map(|id| batcher.metrics().iter().find(|m| m.id == *id))
            .map(|m| m.gen_len)
            .sum();
        let uptime = machine_state();
        eprintln!("[bench] rep {rep} done wall={wall:.2}s tokens={tokens} {uptime}");
        let uptime_s: String = uptime
            .split("uptime_s=")
            .nth(1)
            .and_then(|s| s.split(' ').next())
            .map(|s| s.to_string())
            .unwrap_or_default();
        for id in &ids {
            if let Some(m) = batcher.metrics().iter().find(|m| m.id == *id) {
                let row = format!(
                    "R,{},{},{},{},{},{:.1},{:.1},{},{},,,{}",
                    tag, rep, m.id, m.prompt_len, m.gen_len,
                    m.ttft_us / 1e3, m.tpot_us / 1e3, m.reused_blocks, m.preempted, uptime_s
                );
                println!("{row}");
                if let Some(f) = csv_file.as_mut() {
                    writeln!(f, "{row}").unwrap();
                }
            }
        }
        if let Some(f) = csv_file.as_mut() {
            writeln!(
                f,
                "S,{},{},{},{},{},{},{},{},{},{},{:.2},{},{}",
                tag, rep, "", "", "", "", "", "", "", "", wall, tokens, uptime_s
            )
            .unwrap();
        }
    }
}

/// Real resident-set bytes from /proc/self/statm (Linux); None elsewhere —
/// the registry's accounting is an ESTIMATE and this is its honest
/// cross-check (docs/LIFECYCLE.md gate 4).
fn rss_bytes() -> Option<usize> {
    let s = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: usize = s.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * 4096)
}

fn cmd_models() {
    use inferno_model::registry::ModelRegistry;
    let a = args();
    let dir = flag(&a, "--model-dir", "/root/models/qwen2.5-0.5b");
    let budget_mb: usize = flag(&a, "--budget-mb", "1400").parse().unwrap_or(1400);
    let mut reg = ModelRegistry::new(budget_mb * 1024 * 1024);
    // KV geometry for the demo model (Qwen2.5-0.5B); the registry is
    // file-agnostic — three names, one file, so the lifecycle mechanics
    // (load/evict/reload) are what's being demonstrated, not vocabularies.
    let kv = KvConfig {
        block_size: 16,
        num_blocks: 128,
        num_layers: 24,
        num_kv_heads: 2,
        head_dim: 64,
    };
    for name in ["primary", "secondary", "router"] {
        let d = dir.clone();
        reg.register(name, kv.clone(), Box::new(move || QwenEngine::load(std::path::Path::new(&d))));
    }
    let rss = || -> String {
        match rss_bytes() {
            Some(b) => format!("{} MB", b / 1024 / 1024),
            None => "n/a".into(),
        }
    };
    let show = |step: &str, reg: &ModelRegistry| {
        println!(
            "  {step:<28} resident={} ({} MB) evictions={} rss={}",
            reg.resident_names().join(","),
            reg.resident_bytes() / 1024 / 1024,
            reg.evictions(),
            rss()
        );
    };
    println!(
        "models mode — budget {} MB, names {:?}",
        budget_mb,
        reg.known()
    );
    reg.get("primary").unwrap();
    show("load primary (cold)", &reg);
    reg.get("secondary").unwrap();
    show("load secondary (evicts LRU)", &reg);
    reg.get("primary").unwrap();
    show("reprimary (reload #2)", &reg);
    println!("    primary load_events={} (evict+reload is visible here)", reg.load_events("primary"));
    // Pinned-model safety: an in-flight request must block eviction, loudly.
    reg.begin_request("primary").unwrap();
    let e = match reg.get("secondary") {
        Err(e) => e,
        Ok(_) => panic!("loading secondary must refuse while primary is in flight"),
    };
    println!("  in-flight refusal: {e}");
    show("primary pinned, nothing evicted", &reg);
    reg.end_request("primary");
    reg.get("secondary").unwrap();
    show("release pin, load secondary", &reg);
    println!(
        "  DONE — evictions={} load_events p/s/r = {}/{}/{}",
        reg.evictions(),
        reg.load_events("primary"),
        reg.load_events("secondary"),
        reg.load_events("router")
    );
}

/// Parse --weights (bf16|int8). int8 quantizes once at load
/// (docs/WEIGHTS8.md); bf16 is the exact default.
fn storage_flag(a: &[String]) -> inferno_model::weights::Storage {
    match flag(a, "--weights", "bf16").as_str() {
        "int8" => inferno_model::weights::Storage::Int8,
        other => {
            assert_eq!(other, "bf16", "--weights must be bf16 or int8");
            inferno_model::weights::Storage::Bf16
        }
    }
}

fn cmd_model(admission: std::sync::Arc<dyn Admission>) {
    let a = args();
    let dir = flag(&a, "--model-dir", "/root/models/qwen2.5-0.5b");
    let prompt = flag(&a, "--prompt", "The capital of France is");
    let max_gen: usize = flag(&a, "--max-gen", "24").parse().unwrap_or(24);
    let blocks: usize = flag(&a, "--blocks", "512").parse().unwrap_or(512);

    println!("model mode — loading {dir} ...");
    let storage = storage_flag(&a);
    let mut engine =
        QwenEngine::load_with_storage(std::path::Path::new(&dir), storage).expect("model load failed");
    println!("  storage: {}", storage.tag());
    println!(
        "  model: {} layers | {} q / {} kv heads, head_dim {} | hidden {} | vocab {}",
        engine.cfg.num_layers,
        engine.cfg.num_q_heads,
        engine.cfg.num_kv_heads,
        engine.cfg.head_dim,
        engine.cfg.hidden,
        engine.cfg.vocab
    );

    let prompt_ids = engine.encode_text(&prompt);
    println!("  prompt: {prompt:?} -> {} tokens: {prompt_ids:?}", prompt_ids.len());
    let roundtrip = engine.decode_text(&prompt_ids);
    println!("  round-trip: {roundtrip:?} {}", if roundtrip == prompt { "OK" } else { "MISMATCH <-- tokenizer broken" });
    for &tid in &prompt_ids {
        println!(
            "    tok {tid:>6}  {:?}",
            engine.decode_text(&[tid])
        );
    }

    // forward-path probe: top-5 next-token candidates straight through the
    // engine (no batcher) — separates math bugs from batcher wiring bugs
    let top = engine.diagnostic_topk(&prompt_ids, 5);
    for (tok, lg) in &top {
        println!(
            "  probe: logit {lg:8.3}  tok {tok:>6}  {:?}",
            engine.decode_text(&[*tok])
        );
    }

    let cfg = engine.cfg.clone();
    let mut batcher = Batcher::new(
        KvConfig {
            block_size: 16,
            num_blocks: blocks,
            num_layers: cfg.num_layers,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
        },
        BatcherConfig {
            prefill_chunk_tokens: 64,
            max_num_seqs: 8,
        },
    );
    let _id = batcher.submit(Submit {
        prompt: prompt_ids.clone(),
        max_gen,
        priority: Priority::Normal,
        ttft_slo_ms: 10_000.0,
    });
    // Determinism + retention gate. Two modes:
    //   --repeats N        submit N identical requests up front (concurrent;
    //                      they all reserve before any registration, so
    //                      reuse stays 0 — the determinism gate).
    //   --stagger N        N sequential rounds: submit ONE request, drain it
    //                      to completion, then submit the next. The prefix
    //                      cache persists across rounds, so rounds 2+ must
    //                      show reused_blocks >= 1 (the retention gate).
    let repeats: usize = flag(&a, "--repeats", "1").parse().unwrap_or(1);
    let stagger: bool = flag(&a, "--stagger", "0") == "1";
    let t0 = Instant::now();
    if stagger {
        for r in 0..repeats {
            let id = batcher.submit(Submit {
                prompt: prompt_ids.clone(),
                max_gen,
                priority: Priority::Normal,
                ttft_slo_ms: 10_000.0,
            });
            eprintln!("  [trace] round {r}: submitted id {id}");
            drain_traced(&mut batcher, &mut engine, admission.as_ref(), false);
        }
    } else {
        for _ in 0..repeats {
            batcher.submit(Submit {
                prompt: prompt_ids.clone(),
                max_gen,
                priority: Priority::Normal,
                ttft_slo_ms: 10_000.0,
            });
        }
        drain_traced(&mut batcher, &mut engine, admission.as_ref(), false);
    }
    let wall = t0.elapsed();
    println!("  drained in {:.2} s", wall.as_secs_f64());

    let mut first_ids: Option<&Vec<u32>> = None;
    for (i, toks) in batcher.finished_tokens.iter() {
        if let Some(m) = batcher.metrics().iter().find(|m| m.id == *i) {
            println!(
                "  req {i}: ttft {:.1} ms | tpot {:.1} ms | gen {} tok | reuse {} blk | preempt {}",
                m.ttft_us / 1e3,
                m.tpot_us / 1e3,
                m.gen_len,
                m.reused_blocks,
                m.preempted
            );
        }
        let text = engine.decode_text(toks);
        let same = first_ids.map(|f| f == toks).unwrap_or(true);
        let top = engine.stash_topk(*i, 3).unwrap_or_default();
        println!(
            "  req {i} ids: {:?}\n  req {i} text: {text}{}",
            toks,
            if same { "" } else { "  <-- DETERMINISM FAIL" }
        );
        println!(
            "  req {i} final-stash top3: {:?}",
            top.iter().map(|(t, l)| (format!("{l:.3}"), engine.decode_text(&[*t]))).collect::<Vec<_>>()
        );
        if first_ids.is_none() {
            first_ids = Some(toks);
        }
    }
    if batcher.finished_tokens.is_empty() {
        println!("  WARNING: no finished requests — KV capacity or preemption issue");
    }
}

fn cmd_tok() {
    let tok = tokenizers::Tokenizer::from_file("/root/models/qwen2.5-0.5b/tokenizer.json")
        .expect("tokenizer load failed");
    for text in [
        "The capital of France is",
        "Hello",
        "The",
        " is",
        "Hello world",
    ] {
        let enc = tok.encode(text, true).expect("encode failed");
        println!("{text:?}");
        println!("  ids:    {:?}", enc.get_ids());
        println!("  tokens: {:?}", enc.get_tokens());
        match tok.decode(enc.get_ids(), false) {
            Ok(d) => println!("  decode: {d:?}"),
            Err(e) => println!("  decode: ERROR {e}"),
        }
    }
    let a = tok.encode("The", true).unwrap().get_ids().to_vec();
    let b = tok.encode("The", true).unwrap().get_ids().to_vec();
    println!("same-process twice: {a:?} vs {b:?}");
    println!(
        "id_to_token(785)={:?} id_to_token(9707)={:?} id_to_token(374)={:?}",
        tok.id_to_token(785),
        tok.id_to_token(9707),
        tok.id_to_token(374)
    );
}

fn main() {
    let a = args();
    let mode = a.first().map(|s| s.as_str()).unwrap_or("load");
    let (admission, label) = build_admission();
    match mode {
        "threads" => cmd_threads(),
        "load" => cmd_load(admission, label),
        "overhead" => cmd_overhead(),
        "model" => cmd_model(admission),
        "bench" => cmd_bench(admission, label),
        "models" => cmd_models(),
        "tok" => cmd_tok(),
        _ => {
            eprintln!("usage: inferno-bench [load|threads|model|bench|overhead] [options]");
            eprintln!("  load: --requests N --max-gen N --prompt-min N --prompt-max N --chunk N --seqs N [--tag-interactive]");
            eprintln!("  threads: --threads N");
            eprintln!("  model: --model-dir PATH [--prompt TEXT] [--max-gen N] [--blocks N]");
            eprintln!("  bench: --prompt-len N --max-gen N --conc N --reps N [--stagger 1] --tag NAME --csv PATH");
            eprintln!("  models: [--model-dir PATH] [--budget-mb N] registry lifecycle demo (load/evict/reload)");
            eprintln!("  tok: (no options) direct tokenizers-crate probe");
            std::process::exit(2);
        }
    }
    let _ = std::io::stdout().flush();
}
