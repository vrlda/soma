//! Port of `soma/memory/mixing.py` (CircuitMixingMemory).
//!
//! Growing byte-context circuits under a hard budget with plastic evidence
//! arbitration. Semantics, key layout, integer circuit updates, and the
//! reclamation order match the Python reference exactly; floating point
//! uses the same operation order so parity holds to rounding.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const ORDER_SHIFT: u32 = 56;
const PROBABILITY_FLOOR: f64 = 1.0 / 4096.0;
const STRETCH_LIMIT: f64 = 8.0;

/// Multiply-xorshift hasher for exact u64 keys (keys are not adversarial).
#[derive(Default)]
pub struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        let mut x = self.0;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

type KeyMap<V> = HashMap<u64, V, BuildHasherDefault<KeyHasher>>;

#[derive(Clone, Copy, Debug, Default)]
pub struct Circuit {
    pub n0: u32,
    pub n1: u32,
    pub visits: u32,
    pub last_used: u64,
}

#[derive(Clone, Debug)]
pub struct MixerConfig {
    pub orders: Vec<u32>,
    pub max_circuits: usize,
    pub learning_rate: f64,
    pub count_limit: u32,
    pub initial_weight: f64,
    pub reclaim_fraction: f64,
    pub arbitration: bool,
    /// Opposing counts above this are halved on each update (forgetting).
    pub halve_above: u32,
    /// Plastic calibration: learned probability per (order, n0, n1) state.
    pub calibration: bool,
    pub calibration_limit: u32,
    /// Arbitration weight sets also gated by bit position in the byte.
    pub gate_bit_position: bool,
    /// Arbitration weight sets gated by the partial byte (256 per level).
    pub gate_partial: bool,
    /// Final correction stage over (previous byte, partial byte, evidence).
    pub correction: bool,
    pub correction_rate: f64,
    /// Evidence-gated growth: a circuit above the first order is created
    /// only when the next shorter context's circuit has at least this many
    /// visits (0 = always grow).
    pub growth_threshold: u32,
}

impl Default for MixerConfig {
    fn default() -> Self {
        MixerConfig {
            orders: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
            max_circuits: 1 << 22,
            learning_rate: 0.002,
            count_limit: 1023,
            initial_weight: 0.3,
            reclaim_fraction: 0.125,
            arbitration: true,
            halve_above: 1_000_000,
            calibration: false,
            calibration_limit: 255,
            gate_bit_position: false,
            gate_partial: true,
            correction: true,
            correction_rate: 0.02,
            growth_threshold: 8,
        }
    }
}

pub fn squash(value: f64) -> f64 {
    if value > 40.0 {
        return 1.0;
    }
    if value < -40.0 {
        return 0.0;
    }
    1.0 / (1.0 + (-value).exp())
}

pub fn stretch(probability: f64) -> f64 {
    (probability / (1.0 - probability)).ln()
}

pub const MAX_ORDER: u32 = 32;
const EXACT_ORDER: u32 = 6;
const HASH_MASK: u64 = (1 << 48) - 1;

/// Order in the top byte, the partial byte in the low byte. Contexts of up
/// to six bytes are stored exactly; longer ones as a 48-bit FNV-1a digest.
pub fn circuit_key(order: u32, history: &[u8], partial: u32) -> u64 {
    let start = history.len() - order as usize;
    let context = if order <= EXACT_ORDER {
        let mut context: u64 = 0;
        for &byte in &history[start..] {
            context = (context << 8) | byte as u64;
        }
        context
    } else {
        let mut digest: u64 = 0xcbf29ce484222325;
        for &byte in &history[start..] {
            digest ^= byte as u64;
            digest = digest.wrapping_mul(0x100000001b3);
        }
        digest & HASH_MASK
    };
    ((order as u64) << ORDER_SHIFT) | (context << 8) | partial as u64
}

const CORRECTION_BUCKETS: usize = 33;

fn bit_position(partial: u32) -> usize {
    (31 - partial.leading_zeros()) as usize
}

pub struct CircuitMixingMemory {
    pub config: MixerConfig,
    pub weights: Vec<Vec<f64>>,
    pub circuits: KeyMap<Circuit>,
    pub history: Vec<u8>,
    pub partial: u32,
    pub events_seen: u64,
    pub circuits_created: u64,
    pub circuits_reclaimed: u64,
    /// Per order: (probability, updates) for each (n0, n1) count state.
    pub calibration: Vec<Vec<(f64, u32)>>,
    /// Per (previous byte, partial byte): interpolation table over evidence.
    pub correction: Vec<f64>,
    keys: Vec<Option<u64>>,
    visits: Vec<u32>,
    states: Vec<Option<usize>>,
    inputs: Vec<f64>,
    gate: usize,
    mixed: f64,
    correction_slot: usize,
    correction_mix: f64,
    probability: f64,
    pending: bool,
}

impl CircuitMixingMemory {
    pub fn new(config: MixerConfig) -> Self {
        let n = config.orders.len();
        assert!(n > 0, "orders must be nonempty");
        assert!(config.orders.windows(2).all(|w| w[0] < w[1]), "orders ascending");
        assert!(*config.orders.last().unwrap() <= MAX_ORDER, "order above maximum");
        assert!(config.max_circuits >= 256 * n, "circuit budget is too small");
        let gates = (n + 1) * if config.gate_partial {
            256
        } else if config.gate_bit_position {
            8
        } else {
            1
        };
        let weights = vec![vec![config.initial_weight; n + 1]; gates];
        let side = config.count_limit as usize + 1;
        let calibration = if config.calibration {
            let mut table = Vec::with_capacity(side * side);
            for n0 in 0..side {
                for n1 in 0..side {
                    table.push(((n1 as f64 + 0.4) / (n0 as f64 + n1 as f64 + 0.8), 0u32));
                }
            }
            vec![table; n]
        } else {
            Vec::new()
        };
        let correction = if config.correction {
            let mut row = Vec::with_capacity(CORRECTION_BUCKETS);
            for j in 0..CORRECTION_BUCKETS {
                row.push(squash((j as f64 - 16.0) * 0.5));
            }
            let mut table = Vec::with_capacity(65536 * CORRECTION_BUCKETS);
            for _ in 0..65536 {
                table.extend_from_slice(&row);
            }
            table
        } else {
            Vec::new()
        };
        CircuitMixingMemory {
            weights,
            circuits: KeyMap::default(),
            history: Vec::new(),
            partial: 1,
            events_seen: 0,
            circuits_created: 0,
            circuits_reclaimed: 0,
            calibration,
            correction,
            keys: vec![None; n],
            visits: vec![0; n],
            states: vec![None; n],
            inputs: vec![0.0; n + 1],
            gate: 0,
            mixed: 0.5,
            correction_slot: 0,
            correction_mix: 0.0,
            probability: 0.5,
            pending: false,
            config,
        }
    }

    pub fn reset_history(&mut self) {
        self.history.clear();
        self.partial = 1;
        self.pending = false;
    }

    pub fn predict(&mut self) -> f64 {
        let n = self.config.orders.len();
        let side = self.config.count_limit as usize + 1;
        let mut present = 0;
        for (index, &order) in self.config.orders.iter().enumerate() {
            let key = if order as usize <= self.history.len() {
                Some(circuit_key(order, &self.history, self.partial))
            } else {
                None
            };
            self.keys[index] = key;
            self.states[index] = None;
            self.visits[index] = 0;
            let value = match key.and_then(|k| self.circuits.get(&k)) {
                None => 0.0,
                Some(circuit) => {
                    present += 1;
                    self.visits[index] = circuit.visits;
                    let p = if self.config.calibration {
                        let state = circuit.n0 as usize * side + circuit.n1 as usize;
                        self.states[index] = Some(state);
                        self.calibration[index][state].0
                            .max(PROBABILITY_FLOOR)
                            .min(1.0 - PROBABILITY_FLOOR)
                    } else {
                        (circuit.n1 as f64 + 0.4) / (circuit.n0 as f64 + circuit.n1 as f64 + 0.8)
                    };
                    stretch(p).min(STRETCH_LIMIT).max(-STRETCH_LIMIT)
                }
            };
            self.inputs[index] = value;
        }
        self.inputs[n] = 1.0;
        let mut total = 0.0;
        if self.config.arbitration {
            self.gate = if self.config.gate_partial {
                present * 256 + self.partial as usize
            } else if self.config.gate_bit_position {
                present * 8 + bit_position(self.partial)
            } else {
                present
            };
            let weights = &self.weights[self.gate];
            for index in 0..=n {
                total += weights[index] * self.inputs[index];
            }
        } else {
            for index in (0..n).rev() {
                if self.inputs[index] != 0.0 {
                    total = self.inputs[index];
                    break;
                }
            }
        }
        let mut probability = squash(total).max(PROBABILITY_FLOOR).min(1.0 - PROBABILITY_FLOOR);
        self.mixed = probability;
        if self.config.correction {
            let previous = *self.history.last().unwrap_or(&0) as usize;
            let evidence = stretch(probability).min(STRETCH_LIMIT).max(-STRETCH_LIMIT);
            let position = (evidence + 8.0) * 2.0;
            let mut low = position.floor() as usize;
            if low >= CORRECTION_BUCKETS - 1 {
                low = CORRECTION_BUCKETS - 2;
            }
            let fraction = position - low as f64;
            let slot = ((previous << 8) | self.partial as usize) * CORRECTION_BUCKETS + low;
            let corrected = self.correction[slot] * (1.0 - fraction) + self.correction[slot + 1] * fraction;
            self.correction_slot = slot;
            self.correction_mix = fraction;
            probability = (probability + 3.0 * corrected) / 4.0;
            probability = probability.max(PROBABILITY_FLOOR).min(1.0 - PROBABILITY_FLOOR);
        }
        self.probability = probability;
        self.pending = true;
        probability
    }

    fn reclaim(&mut self) {
        let batch = ((self.config.max_circuits as f64 * self.config.reclaim_fraction) as usize).max(1);
        let mut ranked: Vec<(u32, u64, u64)> = self
            .circuits
            .iter()
            .map(|(&key, c)| (c.visits, c.last_used, key))
            .collect();
        let take = batch.min(ranked.len());
        if take < ranked.len() {
            ranked.select_nth_unstable(take);
        }
        for item in &ranked[..take] {
            self.circuits.remove(&item.2);
        }
        self.circuits_reclaimed += take as u64;
    }

    pub fn observe(&mut self, bit: u32, learn: bool) {
        assert!(bit <= 1, "bit must be 0 or 1");
        if !self.pending {
            self.predict();
        }
        if learn {
            let target = bit as f64;
            if self.config.arbitration {
                let error = target - self.mixed;
                let rate = self.config.learning_rate;
                let weights = &mut self.weights[self.gate];
                for (index, value) in self.inputs.iter().enumerate() {
                    weights[index] += rate * error * value;
                }
            }
            if self.config.calibration {
                let limit = self.config.calibration_limit;
                for index in 0..self.states.len() {
                    if let Some(state) = self.states[index] {
                        let entry = &mut self.calibration[index][state];
                        entry.0 += (target - entry.0) / (entry.1 as f64 + 1.5);
                        if entry.1 < limit {
                            entry.1 += 1;
                        }
                    }
                }
            }
            if self.config.correction {
                let rate = self.config.correction_rate;
                let slot = self.correction_slot;
                let fraction = self.correction_mix;
                self.correction[slot] += (target - self.correction[slot]) * rate * (1.0 - fraction);
                self.correction[slot + 1] += (target - self.correction[slot + 1]) * rate * fraction;
            }
            for index in 0..self.keys.len() {
                let key = match self.keys[index] {
                    None => continue,
                    Some(k) => k,
                };
                if !self.circuits.contains_key(&key) {
                    let threshold = self.config.growth_threshold;
                    if threshold > 0 && index > 0 && self.visits[index - 1] < threshold {
                        continue;
                    }
                    if self.circuits.len() >= self.config.max_circuits {
                        self.reclaim();
                    }
                    self.circuits.insert(key, Circuit::default());
                    self.circuits_created += 1;
                }
                let limit = self.config.count_limit;
                let halve_above = self.config.halve_above;
                let step = self.events_seen;
                let circuit = self.circuits.get_mut(&key).unwrap();
                let (own, other) = if bit == 1 {
                    (&mut circuit.n1, &mut circuit.n0)
                } else {
                    (&mut circuit.n0, &mut circuit.n1)
                };
                if *own < limit {
                    *own += 1;
                }
                if *other > halve_above {
                    *other = (*other + 1) / 2;
                }
                circuit.visits = circuit.visits.saturating_add(1);
                circuit.last_used = step;
            }
        }
        self.partial = (self.partial << 1) | bit;
        if self.partial >= 256 {
            self.history.push((self.partial & 0xFF) as u8);
            let keep = (*self.config.orders.last().unwrap() as usize).max(1);
            if self.history.len() > keep {
                let drop = self.history.len() - keep;
                self.history.drain(..drop);
            }
            self.partial = 1;
        }
        self.events_seen += 1;
        self.pending = false;
    }
    pub fn observe_bytes(&mut self, data: &[u8], learn: bool) {
        for &byte in data {
            for shift in (0..8).rev() {
                self.predict();
                self.observe(((byte >> shift) & 1) as u32, learn);
            }
        }
    }

    /// Mean next-bit cross-entropy in bits/bit; optional probability trace.
    pub fn score_bytes(&mut self, data: &[u8], learn: bool, trace: Option<&mut Vec<f64>>) -> f64 {
        let mut total = 0.0;
        let mut count: u64 = 0;
        let mut trace = trace;
        for &byte in data {
            for shift in (0..8).rev() {
                let bit = ((byte >> shift) & 1) as u32;
                let p = self.predict();
                if let Some(t) = trace.as_deref_mut() {
                    t.push(p);
                }
                total -= (if bit == 1 { p } else { 1.0 - p }).log2();
                count += 1;
                self.observe(bit, learn);
            }
        }
        total / (count.max(1) as f64)
    }
}
