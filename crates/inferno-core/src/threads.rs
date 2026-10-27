//! Worker pool — the "easy accessible threads" pillar.
//!
//! Design notes:
//! * Coarse-grained tasks only (prefill chunks, decode batches, tokenizer
//!   passes). A full work-stealing deque (crossbeam-deque) is overkill for v1:
//!   our tasks are ~100us-10ms, so a single shared queue with a condvar is
//!   within a few percent of optimal and is ~100 lines we can actually verify.
//! * spawn() returns a JoinHandle so callers can await results — the pool is
//!   usable from both sync and async contexts.
//! * Threads are named `inferno_wN` so foresight's classification daemon and
//!   our own sched::Admission client can find and tag them.
//!
//! Shutdown correctness (unit-test-proven): workers sleep on the shared
//! queue's condvar and must still observe shutdown. The queue lock and the
//! wait are atomic (Condvar::wait_while releases the lock while parked), so
//! a notify can never be lost between "empty?" and "wait"; Drop sets the
//! shutdown flag UNDER THE SAME LOCK and notifies all.

use std::collections::VecDeque;
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Shared {
    queue: Mutex<Queue>,
    cond: Condvar,
}

struct Queue {
    jobs: VecDeque<Job>,
    shutdown: bool,
}

pub struct ThreadPool {
    shared: Arc<Shared>,
    handles: Vec<thread::JoinHandle<()>>,
    workers: usize,
}

/// Handle for a spawned task. Await via `join()`.
pub struct JoinHandle<T> {
    rx: mpsc::Receiver<T>,
}

impl<T: Send + 'static> JoinHandle<T> {
    /// Block until the task result is available.
    pub fn join(self) -> Option<T> {
        self.rx.recv().ok()
    }
    /// Non-blocking poll.
    pub fn try_join(&self) -> Option<T> {
        self.rx.try_recv().ok()
    }
}

impl ThreadPool {
    /// Spawn `threads` workers. Defaults to available parallelism.
    pub fn new(threads: Option<usize>) -> Result<Self, String> {
        let threads =
            threads.unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
        if threads == 0 {
            return Err("ThreadPool: threads must be >= 1".into());
        }
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                jobs: VecDeque::new(),
                shutdown: false,
            }),
            cond: Condvar::new(),
        });

        let mut handles = Vec::with_capacity(threads);
        for id in 0..threads {
            let sh = Arc::clone(&shared);
            let name = format!("inferno_w{}", id);
            let handle = thread::Builder::new()
                .name(name)
                .spawn(move || worker_loop(sh))
                .map_err(|e| format!("ThreadPool: failed to spawn worker {}: {}", id, e))?;
            handles.push(handle);
        }
        Ok(Self {
            shared,
            handles,
            workers: threads,
        })
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    /// Submit a task; get its result back via the JoinHandle.
    pub fn spawn<F, T>(&self, f: F) -> JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        let job: Job = Box::new(move || {
            let out = f();
            let _ = tx.send(out);
        });
        {
            let mut q = self.shared.queue.lock().unwrap();
            if q.shutdown {
                // Pool is shutting down; execute inline so the task is not
                // silently dropped (documented contract: post-drop spawn
                // degrades to run-here).
                drop(q);
                job();
                return JoinHandle { rx };
            }
            q.jobs.push_back(job);
        }
        self.shared.cond.notify_one();
        JoinHandle { rx }
    }

    /// Parallel map over OWNED data chunked across workers (fork-join).
    /// Chunks move by value into workers ('static), results come back in
    /// submission order (chunk order). Used by the bench for pool-scaling
    /// checks and by the HTTP plane for tokenizer fan-out.
    pub fn map_chunks<T, R, F>(&self, items: Vec<T>, f: F) -> Vec<R>
    where
        T: Send + 'static,
        R: Send + 'static,
        F: Fn(Vec<T>, usize) -> R + Send + Sync + 'static,
    {
        let f = Arc::new(f);
        let workers = self.workers.max(1);
        let chunk = items.len().div_ceil(workers).max(1);
        let mut handles = Vec::new();
        let mut rest = items;
        let mut chunk_id = 0usize;
        while !rest.is_empty() {
            let take = chunk.min(rest.len());
            let piece: Vec<T> = rest.drain(..take).collect();
            let f = Arc::clone(&f);
            let cid = chunk_id;
            handles.push(self.spawn(move || f(piece, cid)));
            chunk_id += 1;
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("map_chunks task panicked"))
            .collect()
    }

    /// Number of queued jobs (observability; bench verifies drain).
    pub fn pending(&self) -> usize {
        self.shared.queue.lock().unwrap().jobs.len()
    }

    /// Drain: run any queued jobs on the CALLING thread (used by tests and
    /// by the load generator before shutdown when workers are busy).
    pub fn run_queued_inline(&self) -> usize {
        let mut ran = 0;
        loop {
            let job = {
                let mut q = self.shared.queue.lock().unwrap();
                q.jobs.pop_front()
            };
            match job {
                Some(j) => {
                    j();
                    ran += 1;
                }
                None => return ran,
            }
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        {
            let mut q = self.shared.queue.lock().unwrap();
            q.shutdown = true;
        }
        self.shared.cond.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn worker_loop(shared: Arc<Shared>) {
    loop {
        let mut guard = shared.queue.lock().unwrap();
        guard = shared
            .cond
            .wait_while(guard, |q| q.jobs.is_empty() && !q.shutdown)
            .unwrap();
        // wait_while returned => queue has work, or shutdown was set.
        match guard.jobs.pop_front() {
            Some(j) => {
                drop(guard);
                j();
            }
            None => {
                // Only reachable when shutdown && empty.
                debug_assert!(guard.shutdown);
                return;
            }
        }
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::new(None).expect("ThreadPool::default with available_parallelism")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    #[test]
    fn pool_computes_and_returns() {
        let pool = ThreadPool::new(Some(4)).unwrap();
        let h = pool.spawn(|| 40 + 2);
        assert_eq!(h.join(), Some(42));
    }

    #[test]
    fn pool_runs_many_tasks() {
        let pool = ThreadPool::new(Some(4)).unwrap();
        let counter = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::new();
        for _ in 0..1000 {
            let c = Arc::clone(&counter);
            handles.push(pool.spawn(move || {
                c.fetch_add(1, Ordering::Relaxed);
            }));
        }
        for h in handles {
            h.join();
        }
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
        assert_eq!(pool.pending(), 0);
    }

    #[test]
    fn map_chunks_visits_every_item_once() {
        let pool = ThreadPool::new(Some(3)).unwrap();
        let items: Vec<u64> = (0..1001).collect();
        let mut got: Vec<u64> = pool
            .map_chunks(items, |chunk, _id| chunk)
            .into_iter()
            .flatten()
            .collect();
        got.sort_unstable();
        let expect: Vec<u64> = (0..1001).collect();
        assert_eq!(got, expect);
    }

    #[test]
    fn tasks_drain_after_submitter_sleeps() {
        let pool = ThreadPool::new(Some(2)).unwrap();
        let n = Arc::new(AtomicU64::new(0));
        for i in 0..50 {
            let n2 = Arc::clone(&n);
            pool.spawn(move || {
                n2.fetch_add(i + 1, Ordering::Relaxed);
                thread::sleep(Duration::from_micros(100));
            });
        }
        thread::sleep(Duration::from_millis(300));
        assert_eq!(n.load(Ordering::Relaxed), 1275);
        assert_eq!(pool.pending(), 0);
    }

    /// Regression: Drop must not hang when workers are parked on the condvar.
    /// (This is the deadlock the first draft had — scoped guard so Drop runs
    /// while workers are idle-and-parked.)
    #[test]
    fn drop_after_idle_does_not_hang() {
        {
            let _pool = ThreadPool::new(Some(4)).unwrap();
            thread::sleep(Duration::from_millis(50)); // workers all parked
        }
        // if we got here, Drop did not hang
    }

    /// Regression: spawn racing with shutdown must not lose the task.
    #[test]
    fn spawn_during_shutdown_executes_inline() {
        let result_seen;
        {
            let pool = ThreadPool::new(Some(2)).unwrap();
            let h = pool.spawn(|| 7);
            assert_eq!(h.join(), Some(7));
            // simulate shutdown in progress: mark flag directly, then spawn
            {
                let mut q = pool.shared.queue.lock().unwrap();
                q.shutdown = true;
            }
            let h2 = pool.spawn(|| 11);
            result_seen = h2.join() == Some(11);
        }
        assert!(result_seen, "inline execution during shutdown must still return");
    }
}
