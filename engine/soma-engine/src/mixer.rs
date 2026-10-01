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
    /// Metaplasticity: a weight set's rate is learning_rate * tau / (tau + n)
    /// after n updates of that set (0 = constant rate).
    pub plasticity_tau: f64,
    /// Diagnostic: arbitration weights stop learning after this many
    /// events (0 = never).
    pub freeze_arbitration_after: u64,
    /// Pressure-adaptive growth (0 = off): once reclamation has begun, a new
    /// circuit above the first order also needs parent visits >= this *
    /// (frontier + 1), where frontier is the most visits of any circuit in
    /// the last reclaimed batch.
    pub growth_pressure: u32,
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
            correction: false,
            correction_rate: 0.02,
            growth_threshold: 8,
            plasticity_tau: 100_000.0,
            freeze_arbitration_after: 0,
            growth_pressure: 0,
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
pub const STATE_MAGIC: &[u8; 8] = b"SOMAMIX1";
const STATE_VERSION: u64 = 1;

pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut digest: u64 = 0xcbf29ce484222325;
    for &byte in data {
        digest ^= byte as u64;
        digest = digest.wrapping_mul(0x100000001b3);
    }
    digest
}

fn default_correction_row() -> Vec<f64> {
    (0..CORRECTION_BUCKETS).map(|j| squash((j as f64 - 16.0) * 0.5)).collect()
}

struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        if self.offset + count > self.data.len() {
            return Err("SOMAMIX1 truncated".to_string());
        }
        let slice = &self.data[self.offset..self.offset + count];
        self.offset += count;
        Ok(slice)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

fn bit_position(partial: u32) -> usize {
    (31 - partial.leading_zeros()) as usize
}

pub struct CircuitMixingMemory {
    pub config: MixerConfig,
    pub weights: Vec<Vec<f64>>,
    pub weight_updates: Vec<u64>,
    pub circuits: KeyMap<Circuit>,
    pub history: Vec<u8>,
    pub partial: u32,
    pub events_seen: u64,
    pub circuits_created: u64,
    pub circuits_reclaimed: u64,
    /// Most visits of any circuit in the last reclaimed batch.
    pub reclaim_frontier: u32,
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
            weight_updates: vec![0; weights.len()],
            weights,
            circuits: KeyMap::default(),
            history: Vec::new(),
            partial: 1,
            events_seen: 0,
            circuits_created: 0,
            circuits_reclaimed: 0,
            reclaim_frontier: 0,
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
        let mut frontier = 0;
        for item in &ranked[..take] {
            frontier = frontier.max(item.0);
            self.circuits.remove(&item.2);
        }
        if take > 0 {
            self.reclaim_frontier = frontier;
        }
        self.circuits_reclaimed += take as u64;
    }

    pub fn observe(&mut self, bit: u32, learn: bool) {
        self.observe_weighted(bit, learn, 1);
    }

    /// `weight` scales only the count increment (trusted/approved sources);
    /// arbitration and calibration update once, as in the Python reference.
    pub fn observe_weighted(&mut self, bit: u32, learn: bool, weight: u32) {
        assert!(bit <= 1, "bit must be 0 or 1");
        assert!(weight >= 1, "weight must be positive");
        if !self.pending {
            self.predict();
        }
        if learn {
            let target = bit as f64;
            let frozen = self.config.freeze_arbitration_after > 0
                && self.events_seen >= self.config.freeze_arbitration_after;
            if self.config.arbitration && !frozen {
                let error = target - self.mixed;
                let tau = self.config.plasticity_tau;
                let updates = self.weight_updates[self.gate];
                self.weight_updates[self.gate] = updates + 1;
                let rate = if tau > 0.0 {
                    self.config.learning_rate * tau / (tau + updates as f64)
                } else {
                    self.config.learning_rate
                };
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
                    let pressure = self.config.growth_pressure as u64;
                    if pressure > 0
                        && index > 0
                        && self.circuits_reclaimed > 0
                        && (self.visits[index - 1] as u64) < pressure * (self.reclaim_frontier as u64 + 1)
                    {
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
                *own = own.saturating_add(weight).min(limit);
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
    /// (P(next bit = 1), evidence order in bits), as `CircuitMixingMemory.distribution`:
    /// order is the longest live context's whole bytes times 8 plus the bits
    /// already seen of the current byte (0 when no circuit is live).
    pub fn distribution(&mut self) -> (f64, u32) {
        let probability = self.predict();
        let mut order = 0;
        for index in (0..self.config.orders.len()).rev() {
            if self.visits[index] > 0 {
                order = self.config.orders[index] * 8 + bit_position(self.partial) as u32;
                break;
            }
        }
        (probability, order)
    }

    /// Canonical binary state, byte-identical to `CircuitMixingMemory.dumps`.
    pub fn dumps(&self) -> Vec<u8> {
        let c = &self.config;
        let header = serde_json::json!({
            "version": STATE_VERSION,
            "orders": c.orders,
            "max_circuits": c.max_circuits,
            "count_limit": c.count_limit,
            "halve_above": c.halve_above,
            "calibration_limit": c.calibration_limit,
            "growth_threshold": c.growth_threshold,
            "freeze_arbitration_after": c.freeze_arbitration_after,
            "arbitration": c.arbitration,
            "calibration": c.calibration,
            "gate_bit_position": c.gate_bit_position,
            "gate_partial": c.gate_partial,
            "correction": c.correction,
            "events_seen": self.events_seen,
            "circuits_created": self.circuits_created,
            "circuits_reclaimed": self.circuits_reclaimed,
            "partial": self.partial,
            "history": self.history,
        });
        let mut header = header;
        if c.growth_pressure > 0 {
            header["growth_pressure"] = serde_json::json!(c.growth_pressure);
            header["reclaim_frontier"] = serde_json::json!(self.reclaim_frontier);
        }
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(STATE_MAGIC);
        out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
        out.extend_from_slice(&encoded);
        for value in [c.learning_rate, c.initial_weight, c.reclaim_fraction, c.correction_rate, c.plasticity_tau] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let rows = self.weights.len();
        let cols = self.weights[0].len();
        out.extend_from_slice(&(rows as u32).to_le_bytes());
        out.extend_from_slice(&(cols as u32).to_le_bytes());
        for row in &self.weights {
            for value in row {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        for updates in &self.weight_updates {
            out.extend_from_slice(&updates.to_le_bytes());
        }
        let mut keys: Vec<u64> = self.circuits.keys().copied().collect();
        keys.sort_unstable();
        out.extend_from_slice(&(keys.len() as u64).to_le_bytes());
        for key in keys {
            let circuit = &self.circuits[&key];
            out.extend_from_slice(&key.to_le_bytes());
            out.extend_from_slice(&circuit.n0.to_le_bytes());
            out.extend_from_slice(&circuit.n1.to_le_bytes());
            out.extend_from_slice(&circuit.visits.to_le_bytes());
            out.extend_from_slice(&circuit.last_used.to_le_bytes());
        }
        let side = c.count_limit as usize + 1;
        let mut calibration = Vec::new();
        for (index, table) in self.calibration.iter().enumerate() {
            for (state, entry) in table.iter().enumerate() {
                if entry.1 > 0 {
                    calibration.push((index as u32, (state / side) as u32, (state % side) as u32, entry.0, entry.1));
                }
            }
        }
        out.extend_from_slice(&(calibration.len() as u64).to_le_bytes());
        for (index, n0, n1, p, count) in calibration {
            out.extend_from_slice(&index.to_le_bytes());
            out.extend_from_slice(&n0.to_le_bytes());
            out.extend_from_slice(&n1.to_le_bytes());
            out.extend_from_slice(&p.to_le_bytes());
            out.extend_from_slice(&count.to_le_bytes());
        }
        let default_row = default_correction_row();
        let mut rows_out = Vec::new();
        if c.correction {
            for slot in 0..65536usize {
                let row = &self.correction[slot * CORRECTION_BUCKETS..(slot + 1) * CORRECTION_BUCKETS];
                if row != default_row.as_slice() {
                    rows_out.push((slot as u32, row));
                }
            }
        }
        out.extend_from_slice(&(rows_out.len() as u64).to_le_bytes());
        for (slot, row) in rows_out {
            out.extend_from_slice(&slot.to_le_bytes());
            for value in row {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        let checksum = fnv1a64(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    }

    pub fn loads(data: &[u8]) -> Result<Self, String> {
        if data.len() < 20 || &data[..8] != STATE_MAGIC {
            return Err("not a SOMAMIX1 state".to_string());
        }
        let (body, footer) = data.split_at(data.len() - 8);
        if u64::from_le_bytes(footer.try_into().unwrap()) != fnv1a64(body) {
            return Err("SOMAMIX1 checksum mismatch".to_string());
        }
        let mut reader = Reader { data: body, offset: 8 };
        let length = reader.u32()? as usize;
        let header: serde_json::Value =
            serde_json::from_slice(reader.take(length)?).map_err(|e| e.to_string())?;
        let int = |name: &str| header.get(name).and_then(|v| v.as_u64()).ok_or(format!("missing {}", name));
        let flag = |name: &str| header.get(name).and_then(|v| v.as_bool()).ok_or(format!("missing {}", name));
        if int("version")? != STATE_VERSION {
            return Err("unsupported SOMAMIX1 version".to_string());
        }
        let orders: Vec<u32> = header["orders"].as_array().ok_or("missing orders")?
            .iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect();
        let config = MixerConfig {
            orders,
            max_circuits: int("max_circuits")? as usize,
            learning_rate: reader.f64()?,
            count_limit: int("count_limit")? as u32,
            initial_weight: reader.f64()?,
            reclaim_fraction: reader.f64()?,
            arbitration: flag("arbitration")?,
            halve_above: int("halve_above")? as u32,
            calibration: flag("calibration")?,
            calibration_limit: int("calibration_limit")? as u32,
            gate_bit_position: flag("gate_bit_position")?,
            gate_partial: flag("gate_partial")?,
            correction: flag("correction")?,
            correction_rate: reader.f64()?,
            growth_threshold: int("growth_threshold")? as u32,
            plasticity_tau: reader.f64()?,
            freeze_arbitration_after: int("freeze_arbitration_after")?,
            growth_pressure: 0,
        };
        let mut memory = CircuitMixingMemory::new(config);
        let rows = reader.u32()? as usize;
        let cols = reader.u32()? as usize;
        if rows != memory.weights.len() || cols != memory.weights[0].len() {
            return Err("SOMAMIX1 weight shape mismatch".to_string());
        }
        for row in memory.weights.iter_mut() {
            for value in row.iter_mut() {
                *value = reader.f64()?;
            }
        }
        for updates in memory.weight_updates.iter_mut() {
            *updates = reader.u64()?;
        }
        let count = reader.u64()? as usize;
        memory.circuits.reserve(count);
        for _ in 0..count {
            let key = reader.u64()?;
            let circuit = Circuit {
                n0: reader.u32()?,
                n1: reader.u32()?,
                visits: reader.u32()?,
                last_used: reader.u64()?,
            };
            memory.circuits.insert(key, circuit);
        }
        let side = memory.config.count_limit as usize + 1;
        let count = reader.u64()? as usize;
        for _ in 0..count {
            let index = reader.u32()? as usize;
            let n0 = reader.u32()? as usize;
            let n1 = reader.u32()? as usize;
            let p = reader.f64()?;
            let entries = reader.u32()?;
            let table = memory.calibration.get_mut(index).ok_or("calibration index out of range")?;
            let entry = table.get_mut(n0 * side + n1).ok_or("calibration state out of range")?;
            *entry = (p, entries);
        }
        let count = reader.u64()? as usize;
        for _ in 0..count {
            let slot = reader.u32()? as usize;
            if slot >= 65536 || memory.correction.is_empty() {
                return Err("correction row out of range".to_string());
            }
            for j in 0..CORRECTION_BUCKETS {
                memory.correction[slot * CORRECTION_BUCKETS + j] = reader.f64()?;
            }
        }
        if reader.offset != body.len() {
            return Err("SOMAMIX1 trailing bytes".to_string());
        }
        memory.events_seen = int("events_seen")?;
        memory.reclaim_frontier = header.get("reclaim_frontier").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        memory.circuits_created = int("circuits_created")?;
        memory.circuits_reclaimed = int("circuits_reclaimed")?;
        memory.partial = int("partial")? as u32;
        memory.history = header["history"].as_array().ok_or("missing history")?
            .iter().map(|v| v.as_u64().unwrap_or(0) as u8).collect();
        Ok(memory)
    }

    pub fn observe_bytes_weighted(&mut self, data: &[u8], learn: bool, weight: u32) {
        for &byte in data {
            for shift in (0..8).rev() {
                self.predict();
                self.observe_weighted(((byte >> shift) & 1) as u32, learn, weight);
            }
        }
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
