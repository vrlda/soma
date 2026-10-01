//! Differential tape runner: deterministic op sequence from (seed, n).
//!
//! The LCG below is mirrored exactly by engine/differential_tape.py, so both
//! sides execute identical operation tapes without a shared parser.

use soma_engine::tape::{report_inner, run_tape, TapeOp};
use std::time::Instant;

fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) & 0x7fffffff
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let count: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10000);
    let mut state = seed;
    let mut ops = Vec::with_capacity(count);
    for _ in 0..count {
        let kind = lcg(&mut state) % 10;
        let source = lcg(&mut state) % 16;
        let mut destination = lcg(&mut state) % 16;
        if destination == source {
            destination = (destination + 1) % 16;
        }
        if kind < 7 {
            let strength = (lcg(&mut state) as f64 / 2147483648.0) * 2.0 - 1.0;
            ops.push(TapeOp::Add { source, destination, strength });
        } else {
            ops.push(TapeOp::Remove { source, destination });
        }
    }
    let started = Instant::now();
    let graph = run_tape(&ops);
    let elapsed = started.elapsed();
    println!(
        "{{{},\"ops\":{},\"microseconds\":{}}}",
        report_inner(&graph),
        count,
        elapsed.as_micros()
    );
}
