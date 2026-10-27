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

fn main() {
    let a = args();
    let mode = a.first().map(|s| s.as_str()).unwrap_or("load");
    let (admission, label) = build_admission();
    match mode {
        "threads" => cmd_threads(),
        "load" => cmd_load(admission, label),
        "overhead" => cmd_overhead(),
        _ => {
            eprintln!("usage: inferno-bench [load|threads] [options]");
            eprintln!("  load: --requests N --max-gen N --prompt-min N --prompt-max N --chunk N --seqs N [--tag-interactive]");
            eprintln!("  threads: --threads N");
            std::process::exit(2);
        }
    }
    let _ = std::io::stdout().flush();
}
