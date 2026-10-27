//! Continuous batching engine — iteration-level scheduling (Orca, Yu et al.,
//! OSDI 2022) with chunked prefill (Sarathi-Serve, arXiv:2403.02310), SLO-
//! aware ordering, and recompute-preemption on KV exhaustion (vLLM,
//! arXiv:2309.06180 §4.5 Scheduling and Preemption).
//!
//! One `step()` is one scheduler iteration:
//!   1. ADMIT waiting sequences into the running set (priority order),
//!      reserving KV blocks up-front so admission decisions are real.
//!   2. PREFILL: spend the per-step token budget across admitted sequences
//!      (chunked — a long prompt may straddle multiple steps; running decodes
//!      are never stalled waiting for it).
//!   3. DECODE: every running decode sequence advances exactly one token
//!      (iteration-level: new work joins mid-flight).
//!   4. RETIRE finished sequences into metrics; preempt (release + requeue)
//!      any sequence the KV pool can no longer hold.
//!
//! TTFT = arrival → first token; TPOT = (finish − first token) / (gen − 1).

use std::mem;
use std::time::Instant;

use crate::kv_cache::{prefix_hashes, KvCache, KvConfig};
use crate::sched::Admission;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeqState {
    Waiting,
    Prefill,
    Decode,
    Finished,
}

#[derive(Debug)]
pub struct Sequence {
    pub id: u64,
    pub prompt: Vec<u32>,
    /// Prompt tokens processed so far (chunked-prefill cursor).
    pub pos: usize,
    pub generated: Vec<u32>,
    pub max_gen: usize,
    pub state: SeqState,
    pub priority: Priority,
    pub ttft_slo_ms: f32,
    pub arrival: Instant,
    pub first_token_at: Option<Instant>,
    pub finish_at: Option<Instant>,
    /// Simulated forward-pass cost (pseudo engine); real engines report
    /// actual backend time instead.
    pub simulated_compute_us: f64,
    pub queue_wait_us: f64,
    pub reused_blocks: usize,
    pub preempted_count: usize,
}

impl Sequence {
    pub fn prompt_len(&self) -> usize {
        self.prompt.len()
    }
    /// Absolute position the NEXT decode token will occupy.
    pub fn decode_pos(&self) -> usize {
        self.prompt.len() + self.generated.len()
    }
}

/// Engine abstraction: pseudo-model now, real weights backend next PR.
pub trait Engine {
    /// Process at most `budget` more prompt tokens; returns count processed.
    /// Writes K/V through `cache` (paged).
    fn prefill_chunk(&mut self, seq: &mut Sequence, cache: &mut KvCache, budget: usize) -> usize;
    /// One decode iteration for the sequence; returns the token. May update
    /// sequence bookkeeping (simulated cost).
    fn decode_step(&mut self, seq: &mut Sequence, cache: &KvCache) -> u32;
    /// K/V the engine would store for (token, absolute position).
    fn decode_kv(&self, tok: u32, pos: usize) -> Vec<f32>;
    /// EOS token, if the model defines one.
    fn eos(&self) -> Option<u32> {
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BatcherConfig {
    /// Per-step prefill token budget across ALL sequences (Sarathi chunk).
    pub prefill_chunk_tokens: usize,
    /// Max concurrently running sequences.
    pub max_num_seqs: usize,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            prefill_chunk_tokens: 512,
            max_num_seqs: 64,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SeqMetrics {
    pub id: u64,
    pub prompt_len: usize,
    pub gen_len: usize,
    pub queue_wait_us: f64,
    pub ttft_us: f64,
    pub tpot_us: f64,
    pub preempted: usize,
    pub reused_blocks: usize,
    pub priority: Priority,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StepReport {
    pub admitted: usize,
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub finished: usize,
    pub preempted: usize,
    pub kv_free_after: usize,
}

pub struct Submit {
    pub prompt: Vec<u32>,
    pub max_gen: usize,
    pub priority: Priority,
    pub ttft_slo_ms: f32,
}

pub struct Batcher {
    cfg: BatcherConfig,
    cache: KvCache,
    waiting: Vec<Sequence>,
    running: Vec<Sequence>,
    next_seq: u64,
    metrics: Vec<SeqMetrics>,
    total_preemptions: usize,
    finished_ids: Vec<u64>,
}

impl Batcher {
    pub fn new(cache_cfg: KvConfig, cfg: BatcherConfig) -> Self {
        Self {
            cfg,
            cache: KvCache::new(cache_cfg),
            waiting: Vec::new(),
            running: Vec::new(),
            next_seq: 0,
            metrics: Vec::new(),
            total_preemptions: 0,
            finished_ids: Vec::new(),
        }
    }

    pub fn cache_cfg(&self) -> &KvConfig {
        self.cache.cfg()
    }

    pub fn cache_free_blocks(&self) -> usize {
        self.cache.blocks_free()
    }

    pub fn submit(&mut self, req: Submit) -> u64 {
        let id = self.next_seq;
        self.next_seq += 1;
        self.waiting.push(Sequence {
            id,
            prompt: req.prompt,
            pos: 0,
            generated: Vec::new(),
            max_gen: req.max_gen,
            state: SeqState::Waiting,
            priority: req.priority,
            ttft_slo_ms: req.ttft_slo_ms,
            arrival: Instant::now(),
            first_token_at: None,
            finish_at: None,
            simulated_compute_us: 0.0,
            queue_wait_us: 0.0,
            reused_blocks: 0,
            preempted_count: 0,
        });
        id
    }

    pub fn queue_depth(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    pub fn running_ids(&self) -> Vec<u64> {
        self.running.iter().map(|s| s.id).collect()
    }

    pub fn metrics(&self) -> &[SeqMetrics] {
        &self.metrics
    }

    pub fn total_preemptions(&self) -> usize {
        self.total_preemptions
    }

    pub fn finished_ids(&self) -> &[u64] {
        &self.finished_ids
    }

    fn priority_rank(p: Priority) -> u8 {
        p as u8
    }

    fn sort_waiting(waiting: &mut [Sequence]) {
        waiting.sort_by(|a, b| {
            Self::priority_rank(b.priority)
                .cmp(&Self::priority_rank(a.priority))
                .then(
                    a.ttft_slo_ms
                        .partial_cmp(&b.ttft_slo_ms)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(a.arrival.cmp(&b.arrival))
                .then(a.id.cmp(&b.id))
        });
    }

    fn preempt(
        &mut self,
        seq: &mut Sequence,
        cache: &mut KvCache,
        out: &mut StepReport,
    ) {
        cache.release(seq.id);
        seq.pos = 0;
        seq.generated.clear();
        seq.first_token_at = None;
        seq.state = SeqState::Waiting;
        seq.preempted_count += 1;
        self.total_preemptions += 1;
        out.preempted += 1;
    }

    fn finish(&mut self, seq: Sequence, cache: &mut KvCache, out: &mut StepReport) {
        let ttft_us = seq
            .first_token_at
            .map(|t| t.duration_since(seq.arrival).as_secs_f64() * 1e6)
            .unwrap_or(0.0);
        let tpot_us = match (seq.first_token_at, seq.generated.len()) {
            (Some(t), n) if n > 1 => seq
                .finish_at
                .map(|f| f.duration_since(t).as_secs_f64() * 1e6 / (n - 1) as f64)
                .unwrap_or(0.0),
            _ => 0.0,
        };
        self.metrics.push(SeqMetrics {
            id: seq.id,
            prompt_len: seq.prompt.len(),
            gen_len: seq.generated.len(),
            queue_wait_us: seq.queue_wait_us,
            ttft_us,
            tpot_us,
            preempted: seq.preempted_count,
            reused_blocks: seq.reused_blocks,
            priority: seq.priority,
        });
        self.finished_ids.push(seq.id);
        cache.release(seq.id);
        out.finished += 1;
    }

    fn is_done(&self, seq: &Sequence, engine: &dyn Engine, tok: u32) -> bool {
        seq.generated.len() >= seq.max_gen || Some(tok) == engine.eos()
    }

    /// One scheduler iteration. `admission` supplies the foresight-sched
    /// snapshot for the report (NullAdmission when the kernel scheduler is
    /// not loaded).
    pub fn step(&mut self, engine: &mut dyn Engine, admission: &dyn Admission) -> StepReport {
        let mut out = StepReport::default();
        // Take fields out so engine/cache/seq can be borrowed independently.
        let mut running = mem::take(&mut self.running);
        let mut waiting = mem::take(&mut self.waiting);
        let mut cache = mem::take(&mut self.cache);
        let _foresight = admission.snapshot(); // heartbeat; no-op when null

        // --- ADMIT (priority-ordered) ---
        Self::sort_waiting(&mut waiting);
        while !waiting.is_empty() && running.len() < self.cfg.max_num_seqs {
            let bs = cache.cfg().block_size;
            let need = KvCache::blocks_for_tokens(waiting[0].prompt.len(), bs);
            if need > cache.blocks_free() {
                break; // pool can't hold the next prompt; retry next step
            }
            let mut seq = waiting.remove(0);
            let hashes = prefix_hashes(&seq.prompt, bs);
            match cache.reserve(seq.id, seq.prompt.len(), &hashes) {
                Some(info) => {
                    seq.reused_blocks = info.reused;
                    // Shared complete prefix needs no recompute
                    // (RadixAttention, arXiv:2312.07104 — retained KV reuse).
                    seq.pos = (info.reused * bs).min(seq.prompt.len());
                    seq.queue_wait_us = seq.arrival.elapsed().as_secs_f64() * 1e6;
                    seq.state = SeqState::Prefill;
                    running.push(seq);
                    out.admitted += 1;
                }
                None => break,
            }
        }

        // --- PREFILL (chunked budget, priority order) ---
        let mut budget = self.cfg.prefill_chunk_tokens;
        let mut i = 0usize;
        while i < running.len() {
            let seq_state = running[i].state;
            if seq_state != SeqState::Prefill {
                i += 1;
                continue;
            }
            if budget == 0 {
                break;
            }
            let seq = &mut running[i];
            let bs = cache.cfg().block_size;
            let hashes = prefix_hashes(&seq.prompt, bs);
            let pos_before = seq.pos;
            let processed = engine.prefill_chunk(seq, &mut cache, budget);
            budget = budget.saturating_sub(processed);
            let pos_after = seq.pos;
            let prompt_len = seq.prompt.len();
            if processed == 0 && pos_after < prompt_len {
                // Engine refused the chunk (pool/CoW exhausted): preempt.
                let mut seq = running.swap_remove(i);
                self.preempt(&mut seq, &mut cache, &mut out);
                waiting.push(seq);
                continue; // do not advance i
            }
            Self::register_completed_into(&mut cache, seq.id, pos_before, pos_after, &hashes);
            if seq.pos == prompt_len {
                // First token falls out of the completed prompt.
                let need_pos = seq.decode_pos();
                if need_pos >= cache.seq_capacity(seq.id) && !cache.grow(seq.id, 1) {
                    let mut seq = running.swap_remove(i);
                    self.preempt(&mut seq, &mut cache, &mut out);
                    waiting.push(seq);
                    continue;
                }
                let tok = engine.decode_step(seq, &cache);
                seq.first_token_at = Some(Instant::now());
                seq.generated.push(tok);
                let kv = engine.decode_kv(tok, need_pos);
                if cache.write_token(seq.id, need_pos, &kv).is_err() {
                    let mut seq = running.swap_remove(i);
                    self.preempt(&mut seq, &mut cache, &mut out);
                    waiting.push(seq);
                    continue;
                }
                if self.is_done(seq, engine, tok) {
                    seq.state = SeqState::Finished;
                    seq.finish_at = Some(Instant::now());
                } else {
                    seq.state = SeqState::Decode;
                }
            }
            i += 1;
        }

        // --- DECODE (every running decode advances one token) ---
        let mut i = 0usize;
        while i < running.len() {
            let seq_state = running[i].state;
            if seq_state != SeqState::Decode {
                i += 1;
                continue;
            }
            let seq = &mut running[i];
            let need_pos = seq.decode_pos();
            if need_pos >= cache.seq_capacity(seq.id) && !cache.grow(seq.id, 1) {
                let mut seq = running.swap_remove(i);
                self.preempt(&mut seq, &mut cache, &mut out);
                waiting.push(seq);
                continue;
            }
            let tok = engine.decode_step(seq, &cache);
            seq.generated.push(tok);
            let kv = engine.decode_kv(tok, need_pos);
            if cache.write_token(seq.id, need_pos, &kv).is_err() {
                seq.generated.pop(); // roll back before preempt
                let mut seq = running.swap_remove(i);
                self.preempt(&mut seq, &mut cache, &mut out);
                waiting.push(seq);
                continue;
            }
            out.decode_tokens += 1;
            if self.is_done(seq, engine, tok) {
                seq.state = SeqState::Finished;
                seq.finish_at = Some(Instant::now());
            }
            i += 1;
        }

        // --- RETIRE ---
        let mut j = 0usize;
        while j < running.len() {
            if running[j].state == SeqState::Finished {
                let seq = running.swap_remove(j);
                self.finish(seq, &mut cache, &mut out);
            } else {
                j += 1;
            }
        }

        out.kv_free_after = cache.blocks_free();
        self.running = running;
        self.waiting = waiting;
        self.cache = cache;
        out
    }

    fn register_completed_into(
        cache: &mut KvCache,
        seq_id: u64,
        pos_before: usize,
        pos_after: usize,
        hashes: &[u64],
    ) {
        let bs = cache.cfg().block_size;
        for b in (pos_before / bs)..(pos_after / bs) {
            if b < hashes.len() {
                cache.register(seq_id, b, hashes[b]);
            }
        }
    }

    /// Drain everything: run steps until the queue is empty (bench + tests).
    pub fn drain(&mut self, engine: &mut dyn Engine, admission: &dyn Admission) -> Vec<StepReport> {
        let mut reports = Vec::new();
        let mut idle = 0;
        while self.queue_depth() > 0 {
            let r = self.step(engine, admission);
            reports.push(r);
            if r.admitted == 0 && r.prefill_tokens == 0 && r.decode_tokens == 0 && r.preempted == 0
            {
                idle += 1;
                if idle > 1000 {
                    break; // pathological livelock guard
                }
            } else {
                idle = 0;
            }
        }
        reports
    }

    /// Percentile over collected metrics (us). Returns None with no data.
    pub fn percentile_us(&self, p: f64, f: impl Fn(&SeqMetrics) -> f64) -> Option<f64> {
        if self.metrics.is_empty() {
            return None;
        }
        let mut v: Vec<f64> = self.metrics.iter().map(|m| f(m)).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((v.len() as f64) * p).clamp(0.0, v.len() as f64 - 1.0) as usize;
        Some(v[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::KvConfig;
    use crate::pseudo::PseudoEngine;
    use crate::sched::NullAdmission;

    fn seq_cfg() -> (KvConfig, BatcherConfig) {
        (
            KvConfig {
                block_size: 4,
                num_blocks: 64,
                num_layers: 2,
                num_kv_heads: 1,
                head_dim: 4,
            },
            BatcherConfig {
                prefill_chunk_tokens: 64,
                max_num_seqs: 8,
            },
        )
    }

    fn submit_all(b: &mut Batcher, n: usize, prompt_len: usize, max_gen: usize, prio: Priority) {
        for _ in 0..n {
            b.submit(Submit {
                prompt: vec![5; prompt_len],
                max_gen,
                priority: prio,
                ttft_slo_ms: 100.0,
            });
        }
    }

    #[test]
    fn drain_completes_all_and_records_metrics() {
        let (cc, bc) = seq_cfg();
        let mut b = Batcher::new(cc, bc);
        submit_all(&mut b, 5, 12, 6, Priority::Normal);
        let mut eng = PseudoEngine::new(2, 4);
        b.drain(&mut eng, &NullAdmission);
        assert_eq!(b.finished_ids().len(), 5);
        assert_eq!(b.metrics().len(), 5);
        for m in b.metrics() {
            assert_eq!(m.gen_len, 6);
            assert_eq!(m.prompt_len, 12);
            assert_eq!(m.preempted, 0);
        }
        assert_eq!(b.queue_depth(), 0);
        assert_eq!(b.cache_free_blocks(), 64);
    }

    #[test]
    fn chunked_prefill_caps_step_budget() {
        let (cc, _) = seq_cfg();
        let mut b = Batcher::new(
            cc,
            BatcherConfig {
                prefill_chunk_tokens: 5,
                max_num_seqs: 8,
            },
        );
        submit_all(&mut b, 2, 12, 3, Priority::Normal);
        let mut eng = PseudoEngine::new(2, 4);
        let r1 = b.step(&mut eng, &NullAdmission);
        assert_eq!(r1.admitted, 2);
        assert!(r1.prefill_tokens <= 5, "budget must cap the step");
        b.drain(&mut eng, &NullAdmission);
        assert_eq!(b.finished_ids().len(), 2);
        for m in b.metrics() {
            assert_eq!(m.gen_len, 3);
        }
    }

    #[test]
    fn high_priority_admitted_first() {
        let (cc, _) = seq_cfg();
        let mut b = Batcher::new(
            cc,
            BatcherConfig {
                prefill_chunk_tokens: 4,
                max_num_seqs: 1, // one at a time
            },
        );
        b.submit(Submit {
            prompt: vec![1; 8],
            max_gen: 2,
            priority: Priority::Normal,
            ttft_slo_ms: 100.0,
        });
        b.submit(Submit {
            prompt: vec![2; 8],
            max_gen: 2,
            priority: Priority::High,
            ttft_slo_ms: 100.0,
        });
        let mut eng = PseudoEngine::new(2, 4);
        let r = b.step(&mut eng, &NullAdmission);
        assert_eq!(r.admitted, 1);
        assert_eq!(b.running_ids(), vec![1]); // High (id 1) wins
        b.drain(&mut eng, &NullAdmission);
        let order: Vec<u64> = b.metrics().iter().map(|m| m.id).collect();
        assert_eq!(order, vec![1, 0]);
    }

    #[test]
    fn preemption_recovers_when_cache_is_tiny() {
        let (cc, bc) = seq_cfg();
        let mut b = Batcher::new(KvConfig { num_blocks: 6, ..cc }, bc);
        submit_all(&mut b, 4, 8, 4, Priority::Normal); // 4 seqs * 2 blocks = 8 > 6
        let mut eng = PseudoEngine::new(2, 4);
        b.drain(&mut eng, &NullAdmission);
        assert_eq!(b.finished_ids().len(), 4);
        assert!(b.total_preemptions() > 0, "expected at least one preemption");
        for m in b.metrics() {
            assert_eq!(m.gen_len, 4);
        }
    }

    #[test]
    fn duplicate_prompt_gets_prefix_reuse() {
        let (cc, bc) = seq_cfg();
        let mut b = Batcher::new(cc, bc);
        b.submit(Submit {
            prompt: vec![9; 8],
            max_gen: 2,
            priority: Priority::Normal,
            ttft_slo_ms: 1.0,
        });
        let mut eng = PseudoEngine::new(2, 4);
        b.drain(&mut eng, &NullAdmission);
        assert_eq!(b.finished_ids().len(), 1);

        b.submit(Submit {
            prompt: vec![9; 8],
            max_gen: 2,
            priority: Priority::Normal,
            ttft_slo_ms: 1.0,
        });
        let mut eng = PseudoEngine::new(2, 4);
        let r1 = b.step(&mut eng, &NullAdmission);
        assert_eq!(r1.admitted, 1);
        // The duplicate shares both prompt blocks (reused=2 below), needs
        // 1 fresh block for its decode token, and finishes within the step
        // (max_gen=2) — so all blocks are back on the free list after.
        assert_eq!(r1.kv_free_after, 64);
        b.drain(&mut eng, &NullAdmission);
        assert_eq!(b.finished_ids().len(), 2);
        // second metric shows the reuse
        let m2 = &b.metrics()[1];
        assert_eq!(m2.reused_blocks, 2);
    }

    #[test]
    fn percentile_sanity() {
        let (cc, bc) = seq_cfg();
        let mut b = Batcher::new(cc, bc);
        submit_all(&mut b, 3, 4, 2, Priority::Normal);
        let mut eng = PseudoEngine::new(2, 4);
        b.drain(&mut eng, &NullAdmission);
        assert!(b.percentile_us(0.5, |m| m.ttft_us).is_some());
        assert!(b.percentile_us(0.99, |m| m.tpot_us).is_some());
    }
}
