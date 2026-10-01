//! Long-range circuit mixing (ADR 0010, stage 2). Research core.
//!
//! Extends the byte-context circuits of `mixer.rs` with longer-timescale
//! context, all learned locally (counts, a delta rule, plastic arbitration):
//!
//! - byte circuits: the previous 0..12 bytes (as in `mixer.rs`);
//! - word circuits: the current word prefix with the last 1, 2, or 3 words;
//! - skip circuits: the prefix with the word 2 or 3 back (skipping nearer ones);
//! - topic circuit: the prefix with the two most recent content words that
//!   are at least four words back;
//! - line circuit: the prefix with the bag of content words of the previous
//!   line, so a question line can shape the answer line;
//! - match model: the longest earlier repeat of the last bytes in the current
//!   document predicts the byte that followed it, with a confidence learned
//!   per match length.
//!
//! Words are adapter-level: hashed byte spans split at ASCII separators,
//! with no vocabulary. History resets at document boundaries.

use crate::mixer::{squash, stretch};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const PROBABILITY_FLOOR: f64 = 1.0 / 4096.0;
const STRETCH_LIMIT: f64 = 8.0;
const HASH_MASK: u64 = (1 << 48) - 1;
const TYPE_SHIFT: u32 = 56;
const MATCH_MIN: usize = 6;
const MATCH_BUCKETS: usize = 16;
const MAX_DOCUMENT: usize = 1 << 22;
const CONTENT_MIN: u32 = 4;

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

fn mix(seed: u64, value: u64) -> u64 {
    let mut digest = seed ^ 0xcbf29ce484222325;
    for shift in (0..64).step_by(8) {
        digest ^= (value >> shift) & 0xff;
        digest = digest.wrapping_mul(0x100000001b3);
    }
    digest
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Circuit {
    pub n0: u32,
    pub n1: u32,
    pub visits: u32,
    pub last_used: u64,
}

#[derive(Clone, Debug)]
pub struct LongConfig {
    pub byte_orders: Vec<u32>,
    pub word_contexts: bool,
    pub match_model: bool,
    pub max_circuits: usize,
    pub learning_rate: f64,
    pub count_limit: u32,
    pub halve_above: u32,
    pub initial_weight: f64,
    pub reclaim_fraction: f64,
    pub growth_threshold: u32,
    pub growth_pressure: u32,
    pub plasticity_tau: f64,
    /// Weight sets selected by byte-circuit presence only (not word circuits),
    /// and long-range inputs start at weight 0 so they must earn influence.
    pub byte_gate: bool,
    /// Copy pointer for answers (stage 3).
    pub pointer: bool,
}

impl Default for LongConfig {
    fn default() -> Self {
        LongConfig {
            byte_orders: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
            word_contexts: true,
            match_model: true,
            max_circuits: 1 << 22,
            learning_rate: 0.002,
            count_limit: 1023,
            halve_above: 1_000_000,
            initial_weight: 0.3,
            reclaim_fraction: 0.03,
            growth_threshold: 8,
            growth_pressure: 8,
            plasticity_tau: 100_000.0,
            byte_gate: true,
            pointer: true,
        }
    }
}


/// Document-level state (everything that resets with history), for
/// scoring candidate continuations from the same point.
#[derive(Clone)]
pub struct DocState {
    history: Vec<u8>,
    partial: u32,
    word_hash: u64,
    word_len: u32,
    words: [u64; 4],
    content: Vec<u64>,
    content_ages: Vec<u64>,
    words_seen: u64,
    line_bag: Vec<u64>,
    prev_line_bag: u64,
    match_ptr: usize,
    match_len: usize,
    match_table: KeyMap<usize>,
    doc_words: Vec<(u64, usize, bool)>,
    line_start: usize,
    line_words_from: usize,
    question: Option<(Vec<u64>, usize)>,
    answer_line: bool,
    pointer_pos: Option<usize>,
    pointer_score: usize,
    pointer_advance: usize,
}

pub struct LongMixer {
    pub config: LongConfig,
    pub circuits: KeyMap<Circuit>,
    pub weights: Vec<Vec<f64>>,
    pub weight_updates: Vec<u64>,
    // document state
    pub history: Vec<u8>,
    pub partial: u32,
    word_hash: u64,
    word_len: u32,
    words: [u64; 4],
    content: Vec<u64>,
    content_ages: Vec<u64>,
    words_seen: u64,
    line_bag: Vec<u64>,
    prev_line_bag: u64,
    match_ptr: usize,
    match_len: usize,
    match_table: KeyMap<usize>,
    match_counts: [[u32; 2]; MATCH_BUCKETS],
    // copy pointer: words of the document (hash, start, content), question state
    doc_words: Vec<(u64, usize, bool)>,
    line_start: usize,
    line_words_from: usize,
    question: Option<(Vec<u64>, usize)>,
    answer_line: bool,
    pointer_pos: Option<usize>,
    pointer_score: usize,
    pointer_advance: usize,
    pointer_counts: [[u32; 2]; 32],
    pointer_bit: Option<u32>,
    pointer_bucket: usize,
    // per-bit scratch
    keys: Vec<Option<u64>>,
    parents: Vec<Option<usize>>,
    visits: Vec<u32>,
    inputs: Vec<f64>,
    gate: usize,
    mixed: f64,
    match_bit: Option<u32>,
    match_bucket: usize,
    pending: bool,
    // counters
    pub events_seen: u64,
    pub circuits_created: u64,
    pub circuits_reclaimed: u64,
    reclaim_frontier: u32,
}

impl LongMixer {
    pub fn new(config: LongConfig) -> Self {
        let mut parents: Vec<Option<usize>> = Vec::new();
        for index in 0..config.byte_orders.len() {
            parents.push(if index == 0 { None } else { Some(index - 1) });
        }
        let base = config.byte_orders.len();
        if config.word_contexts {
            let order2 = config.byte_orders.iter().position(|&o| o == 2).unwrap_or(0);
            // W1 <- byte order 2; W2 <- W1; W3 <- W2; S2 <- W1; S3 <- S2; TOPIC <- W1; LINE <- W1
            parents.extend([Some(order2), Some(base), Some(base + 1), Some(base), Some(base + 3),
                            Some(base), Some(base)]);
        }
        let circuits_n = parents.len();
        let inputs_n = circuits_n + if config.match_model { 1 } else { 0 }
            + if config.pointer { 1 } else { 0 };
        let rows = (circuits_n + 1) * 3 * 2 * 256;
        let byte_n = config.byte_orders.len();
        let mut initial = vec![config.initial_weight; inputs_n + 1];
        if config.byte_gate {
            for value in initial.iter_mut().take(inputs_n).skip(byte_n) {
                *value = 0.0;
            }
        }
        LongMixer {
            weights: vec![initial; rows],
            weight_updates: vec![0; rows],
            circuits: KeyMap::default(),
            history: Vec::new(),
            partial: 1,
            word_hash: 0,
            word_len: 0,
            words: [0; 4],
            content: Vec::new(),
            content_ages: Vec::new(),
            words_seen: 0,
            line_bag: Vec::new(),
            prev_line_bag: 0,
            match_ptr: 0,
            match_len: 0,
            match_table: KeyMap::default(),
            match_counts: [[0; 2]; MATCH_BUCKETS],
            doc_words: Vec::new(),
            line_start: 0,
            line_words_from: 0,
            question: None,
            answer_line: false,
            pointer_pos: None,
            pointer_score: 0,
            pointer_advance: 0,
            pointer_counts: [[0; 2]; 32],
            pointer_bit: None,
            pointer_bucket: 0,
            keys: vec![None; circuits_n],
            parents,
            visits: vec![0; circuits_n],
            inputs: vec![0.0; inputs_n + 1],
            gate: 0,
            mixed: 0.5,
            match_bit: None,
            match_bucket: 0,
            pending: false,
            events_seen: 0,
            circuits_created: 0,
            circuits_reclaimed: 0,
            reclaim_frontier: 0,
            config,
        }
    }

    pub fn reset_history(&mut self) {
        self.history.clear();
        self.partial = 1;
        self.word_hash = 0;
        self.word_len = 0;
        self.words = [0; 4];
        self.content.clear();
        self.content_ages.clear();
        self.words_seen = 0;
        self.line_bag.clear();
        self.prev_line_bag = 0;
        self.match_ptr = 0;
        self.match_len = 0;
        self.match_table.clear();
        self.doc_words.clear();
        self.line_start = 0;
        self.line_words_from = 0;
        self.question = None;
        self.answer_line = false;
        self.pointer_pos = None;
        self.pending = false;
    }

    fn byte_key(&self, order: u32) -> Option<u64> {
        let order_usize = order as usize;
        if order_usize > self.history.len() {
            return None;
        }
        let start = self.history.len() - order_usize;
        let mut digest: u64 = 0xcbf29ce484222325;
        for &byte in &self.history[start..] {
            digest ^= byte as u64;
            digest = digest.wrapping_mul(0x100000001b3);
        }
        Some(((order as u64) << TYPE_SHIFT) | ((digest & HASH_MASK) << 8) | self.partial as u64)
    }

    fn word_prefix(&self) -> u64 {
        // the current word's hash and length, or the separator just seen
        if self.word_len > 0 {
            mix(self.word_hash, self.word_len as u64)
        } else {
            mix(0x5eb, *self.history.last().unwrap_or(&0) as u64)
        }
    }

    fn word_key(&self, kind: u64, parts: &[u64]) -> u64 {
        let mut digest = mix(kind, self.word_prefix());
        for &part in parts {
            digest = mix(digest, part);
        }
        ((32 + kind) << TYPE_SHIFT) | ((digest & HASH_MASK) << 8) | self.partial as u64
    }

    fn topic_words(&self) -> Option<(u64, u64)> {
        // the two most recent content words at least four words back
        let mut found = Vec::new();
        for index in (0..self.content.len()).rev() {
            if self.words_seen >= self.content_ages[index] + 4 {
                found.push(self.content[index]);
                if found.len() == 2 {
                    break;
                }
            }
        }
        if found.len() == 2 { Some((found[0], found[1])) } else { None }
    }

    pub fn predict(&mut self) -> f64 {
        let mut index = 0;
        for order_index in 0..self.config.byte_orders.len() {
            self.keys[index] = self.byte_key(self.config.byte_orders[order_index]);
            index += 1;
        }
        if self.config.word_contexts {
            let w = self.words;
            let seen = self.words_seen;
            self.keys[index] = if seen >= 1 { Some(self.word_key(1, &[w[0]])) } else { None };
            self.keys[index + 1] = if seen >= 2 { Some(self.word_key(2, &[w[0], w[1]])) } else { None };
            self.keys[index + 2] = if seen >= 3 { Some(self.word_key(3, &[w[0], w[1], w[2]])) } else { None };
            self.keys[index + 3] = if seen >= 2 { Some(self.word_key(4, &[w[1]])) } else { None };
            self.keys[index + 4] = if seen >= 3 { Some(self.word_key(5, &[w[2]])) } else { None };
            self.keys[index + 5] = self.topic_words().map(|(a, b)| self.word_key(6, &[a, b]));
            self.keys[index + 6] = if self.prev_line_bag != 0 {
                let in_line = if self.line_bag.is_empty() { 0 } else { w[0] };
                Some(self.word_key(7, &[self.prev_line_bag, in_line]))
            } else {
                None
            };
        }
        let n = self.keys.len();
        let mut present = 0;
        let byte_n = self.config.byte_orders.len();
        for i in 0..n {
            self.visits[i] = 0;
            self.inputs[i] = match self.keys[i].and_then(|k| self.circuits.get(&k)) {
                None => 0.0,
                Some(circuit) => {
                    if i < byte_n || !self.config.byte_gate {
                        present += 1;
                    }
                    self.visits[i] = circuit.visits;
                    let p = (circuit.n1 as f64 + 0.4) / (circuit.n0 as f64 + circuit.n1 as f64 + 0.8);
                    stretch(p).min(STRETCH_LIMIT).max(-STRETCH_LIMIT)
                }
            };
        }
        let mut match_state = 0;
        self.match_bit = None;
        if self.config.match_model {
            let mut value = 0.0;
            if self.match_len > 0 && self.match_ptr < self.history.len() {
                let expected = self.history[self.match_ptr] as u32 | 256;
                let seen_bits = 32 - self.partial.leading_zeros() - 1; // bits of this byte so far
                if (expected >> (8 - seen_bits)) == self.partial {
                    let bit = (expected >> (7 - seen_bits)) & 1;
                    let bucket = self.match_len.min(MATCH_BUCKETS - 1);
                    let [wrong, right] = self.match_counts[bucket];
                    let p = (right as f64 + 0.5) / (right as f64 + wrong as f64 + 1.0);
                    let confidence = stretch(p).min(STRETCH_LIMIT).max(-STRETCH_LIMIT);
                    value = if bit == 1 { confidence } else { -confidence };
                    self.match_bit = Some(bit);
                    self.match_bucket = bucket;
                    match_state = if self.match_len < 16 { 1 } else { 2 };
                }
            }
            self.inputs[n] = value;
        }
        let mut pointer_state = 0;
        self.pointer_bit = None;
        if self.config.pointer {
            let slot = n + if self.config.match_model { 1 } else { 0 };
            let mut value = 0.0;
            if let Some(position) = self.pointer_pos {
                if position < self.history.len() {
                    let expected = self.history[position] as u32 | 256;
                    let seen_bits = 31 - self.partial.leading_zeros();
                    if (expected >> (8 - seen_bits)) == self.partial {
                        let bit = (expected >> (7 - seen_bits)) & 1;
                        let bucket = self.pointer_score.min(7) * 4 + self.pointer_advance.min(3);
                        let [wrong, right] = self.pointer_counts[bucket];
                        let p = (right as f64 + 0.5) / (right as f64 + wrong as f64 + 1.0);
                        let confidence = stretch(p).min(STRETCH_LIMIT).max(-STRETCH_LIMIT);
                        value = if bit == 1 { confidence } else { -confidence };
                        self.pointer_bit = Some(bit);
                        self.pointer_bucket = bucket;
                        pointer_state = 1;
                    }
                }
            }
            self.inputs[slot] = value;
        }
        let last = self.inputs.len() - 1;
        self.inputs[last] = 1.0;
        self.gate = (((present * 3) + match_state) * 2 + pointer_state) * 256 + self.partial as usize;
        let weights = &self.weights[self.gate];
        let mut total = 0.0;
        for i in 0..self.inputs.len() {
            total += weights[i] * self.inputs[i];
        }
        let probability = squash(total).max(PROBABILITY_FLOOR).min(1.0 - PROBABILITY_FLOOR);
        self.mixed = probability;
        self.pending = true;
        probability
    }

    fn reclaim(&mut self) {
        let batch = ((self.config.max_circuits as f64 * self.config.reclaim_fraction) as usize).max(1);
        let mut ranked: Vec<(u32, u64, u64)> =
            self.circuits.iter().map(|(&key, c)| (c.visits, c.last_used, key)).collect();
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
        if !self.pending {
            self.predict();
        }
        if learn {
            let target = bit as f64;
            let error = target - self.mixed;
            let updates = self.weight_updates[self.gate];
            self.weight_updates[self.gate] = updates + 1;
            let tau = self.config.plasticity_tau;
            let rate = if tau > 0.0 {
                self.config.learning_rate * tau / (tau + updates as f64)
            } else {
                self.config.learning_rate
            };
            let weights = &mut self.weights[self.gate];
            for (i, value) in self.inputs.iter().enumerate() {
                weights[i] += rate * error * value;
            }
            if let Some(expected) = self.match_bit {
                let slot = &mut self.match_counts[self.match_bucket][(expected == bit) as usize];
                *slot = slot.saturating_add(1);
                let pair = self.match_counts[self.match_bucket];
                if pair[0] + pair[1] > 65_535 {
                    self.match_counts[self.match_bucket] = [pair[0] / 2, pair[1] / 2];
                }
            }
            if let Some(expected) = self.pointer_bit {
                let slot = &mut self.pointer_counts[self.pointer_bucket][(expected == bit) as usize];
                *slot = slot.saturating_add(1);
                let pair = self.pointer_counts[self.pointer_bucket];
                if pair[0] + pair[1] > 65_535 {
                    self.pointer_counts[self.pointer_bucket] = [pair[0] / 2, pair[1] / 2];
                }
            }
            let step = self.events_seen;
            for i in 0..self.keys.len() {
                let key = match self.keys[i] {
                    None => continue,
                    Some(k) => k,
                };
                if !self.circuits.contains_key(&key) {
                    if let Some(parent) = self.parents[i] {
                        let parent_visits = self.visits[parent];
                        if parent_visits < self.config.growth_threshold {
                            continue;
                        }
                        let pressure = self.config.growth_pressure as u64;
                        if pressure > 0 && self.circuits_reclaimed > 0
                            && (parent_visits as u64) < pressure * (self.reclaim_frontier as u64 + 1)
                        {
                            continue;
                        }
                    }
                    if self.circuits.len() >= self.config.max_circuits {
                        self.reclaim();
                    }
                    self.circuits.insert(key, Circuit::default());
                    self.circuits_created += 1;
                }
                let limit = self.config.count_limit;
                let circuit = self.circuits.get_mut(&key).unwrap();
                let (own, other) = if bit == 1 {
                    (&mut circuit.n1, &mut circuit.n0)
                } else {
                    (&mut circuit.n0, &mut circuit.n1)
                };
                *own = (*own + 1).min(limit);
                if *other > self.config.halve_above {
                    *other = (*other + 1) / 2;
                }
                circuit.visits = circuit.visits.saturating_add(1);
                circuit.last_used = step;
            }
        }
        self.partial = (self.partial << 1) | bit;
        if self.partial >= 256 {
            let byte = (self.partial & 0xff) as u8;
            self.partial = 1;
            self.complete_byte(byte);
        }
        self.events_seen += 1;
        self.pending = false;
    }

    fn complete_byte(&mut self, byte: u8) {
        if self.history.len() >= MAX_DOCUMENT {
            // very long documents: keep the recent half
            let keep = MAX_DOCUMENT / 2;
            let drop = self.history.len() - keep;
            self.history.drain(..drop);
            self.match_table.clear();
            self.match_len = 0;
        }
        self.history.push(byte);
        if let Some(position) = self.pointer_pos {
            if position < self.history.len() - 1 && self.history[position] == byte {
                self.pointer_pos = Some(position + 1);
                self.pointer_advance += 1;
            } else {
                self.pointer_pos = None;
            }
        }
        // match model: extend or look up
        if self.config.match_model {
            if self.match_len > 0 && self.match_ptr < self.history.len() - 1
                && self.history[self.match_ptr] == byte
            {
                self.match_ptr += 1;
                self.match_len += 1;
            } else {
                self.match_len = 0;
            }
            let length = self.history.len();
            if length >= MATCH_MIN {
                let digest = self.history[length - MATCH_MIN..]
                    .iter()
                    .fold(0xcbf29ce484222325u64, |d, &b| (d ^ b as u64).wrapping_mul(0x100000001b3));
                if self.match_len == 0 {
                    if let Some(&position) = self.match_table.get(&digest) {
                        self.match_ptr = position;
                        self.match_len = 1;
                    }
                }
                self.match_table.insert(digest, length);
            }
        }
        // words
        let lower = byte.to_ascii_lowercase();
        let word_char = lower.is_ascii_alphanumeric() || byte >= 128;
        if word_char {
            self.word_hash = mix(self.word_hash, lower as u64);
            self.word_len += 1;
        } else {
            if self.word_len > 0 {
                let word = mix(self.word_hash, self.word_len as u64);
                if self.config.pointer && self.doc_words.len() < 1 << 20 {
                    let start = self.history.len() - 1 - self.word_len as usize;
                    self.doc_words.push((word, start, self.word_len >= CONTENT_MIN));
                }
                self.words = [word, self.words[0], self.words[1], self.words[2]];
                self.words_seen += 1;
                if self.word_len >= CONTENT_MIN {
                    self.content.push(word);
                    self.content_ages.push(self.words_seen);
                    if self.content.len() > 16 {
                        self.content.remove(0);
                        self.content_ages.remove(0);
                    }
                    if self.line_bag.len() < 24 {
                        self.line_bag.push(word);
                    }
                }
            }
            self.word_hash = 0;
            self.word_len = 0;
            if self.config.pointer && self.answer_line && self.pointer_pos.is_none()
                && byte == b' ' && self.history.len() >= 2
                && self.history[self.history.len() - 2] == b':'
                && self.history.len() - self.line_start <= 16
            {
                self.align_pointer();
            }
            if byte == b'\n' {
                if self.config.pointer {
                    let line = &self.history[self.line_start..self.history.len() - 1];
                    let is_question = line.iter().rev().find(|b| !b.is_ascii_whitespace()) == Some(&b'?');
                    self.answer_line = false;
                    self.pointer_pos = None;
                    if is_question {
                        let words: Vec<u64> = self.doc_words[self.line_words_from..]
                            .iter().filter(|w| w.2).map(|w| w.0).collect();
                        self.question = Some((words, self.line_words_from));
                        self.answer_line = true;
                    }
                    self.line_start = self.history.len();
                    self.line_words_from = self.doc_words.len();
                }
                if !self.line_bag.is_empty() {
                    let mut bag = self.line_bag.clone();
                    bag.sort_unstable();
                    bag.dedup();
                    self.prev_line_bag = bag.iter().fold(0x1b, |d, &w| mix(d, w));
                }
                self.line_bag.clear();
            }
        }
    }

    /// Point at the word in the earlier text with the most question content
    /// words just before it (window of 10 words), skipping question words.
    fn align_pointer(&mut self) {
        let (question, question_from) = match &self.question {
            Some(q) => q.clone(),
            None => return,
        };
        if question.is_empty() {
            return;
        }
        const WINDOW: usize = 10;
        let mut best = (0usize, 0usize);
        for i in 1..question_from.min(self.doc_words.len()) {
            if question.contains(&self.doc_words[i].0) {
                continue;
            }
            let from = i.saturating_sub(WINDOW);
            let mut found: Vec<u64> = Vec::new();
            for w in &self.doc_words[from..i] {
                if question.contains(&w.0) && !found.contains(&w.0) {
                    found.push(w.0);
                }
            }
            if found.len() > best.0 {
                best = (found.len(), i);
            }
        }
        if best.0 > 0 {
            self.pointer_pos = Some(self.doc_words[best.1].1);
            self.pointer_score = best.0;
            self.pointer_advance = 0;
        }
    }

    pub fn snapshot(&self) -> DocState {
        DocState {
            history: self.history.clone(), partial: self.partial, word_hash: self.word_hash,
            word_len: self.word_len, words: self.words, content: self.content.clone(),
            content_ages: self.content_ages.clone(), words_seen: self.words_seen,
            line_bag: self.line_bag.clone(), prev_line_bag: self.prev_line_bag,
            match_ptr: self.match_ptr, match_len: self.match_len,
            match_table: self.match_table.clone(), doc_words: self.doc_words.clone(),
            line_start: self.line_start, line_words_from: self.line_words_from,
            question: self.question.clone(), answer_line: self.answer_line,
            pointer_pos: self.pointer_pos, pointer_score: self.pointer_score,
            pointer_advance: self.pointer_advance,
        }
    }

    pub fn restore(&mut self, state: &DocState) {
        let s = state.clone();
        self.history = s.history; self.partial = s.partial; self.word_hash = s.word_hash;
        self.word_len = s.word_len; self.words = s.words; self.content = s.content;
        self.content_ages = s.content_ages; self.words_seen = s.words_seen;
        self.line_bag = s.line_bag; self.prev_line_bag = s.prev_line_bag;
        self.match_ptr = s.match_ptr; self.match_len = s.match_len;
        self.match_table = s.match_table; self.doc_words = s.doc_words;
        self.line_start = s.line_start; self.line_words_from = s.line_words_from;
        self.question = s.question; self.answer_line = s.answer_line;
        self.pointer_pos = s.pointer_pos; self.pointer_score = s.pointer_score;
        self.pointer_advance = s.pointer_advance;
        self.pending = false;
    }

    /// Log-probability of ``candidate + stop`` from a snapshot (frozen).
    fn continuation_log_probability(&mut self, start: &DocState, candidate: &[u8], stop: u8)
        -> f64
    {
        self.restore(start);
        let mut bytes = candidate.to_vec();
        bytes.push(stop);
        let (log_loss, _) = self.score_bytes(&bytes, false);
        -log_loss
    }

    /// Extractive answer: score every span of 1..=max_words words from
    /// ``passage`` as ``span + stop`` after ``prompt`` (frozen). ``scoring``:
    /// "sum" (total log-probability), "mean" (per byte), or "pmi" (minus the
    /// same span's log-probability after ``neutral``, i.e. how much the
    /// question raises it).
    pub fn best_span(&mut self, prompt: &[u8], neutral: &[u8], passage: &[u8], max_words: usize,
                     stop: u8, scoring: &str) -> Vec<u8>
    {
        self.reset_history();
        self.observe_bytes(prompt, false);
        let start = self.snapshot();
        let neutral_start = if scoring == "pmi" {
            self.reset_history();
            self.observe_bytes(neutral, false);
            Some(self.snapshot())
        } else {
            None
        };
        let text = String::from_utf8_lossy(passage).to_string();
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let mut seen = std::collections::HashSet::new();
        let mut best: (f64, Vec<u8>) = (f64::NEG_INFINITY, Vec::new());
        for i in 0..tokens.len() {
            for j in i + 1..=(i + max_words).min(tokens.len()) {
                let span = tokens[i..j].join(" ");
                let span = span.trim_matches(|c: char| ",.;:!?\"'()".contains(c)).to_string();
                if span.is_empty() || !seen.insert(span.clone()) {
                    continue;
                }
                let bytes = span.as_bytes();
                let mut score = self.continuation_log_probability(&start, bytes, stop);
                match scoring {
                    "mean" => score /= (bytes.len() + 1) as f64,
                    "pmi" => {
                        let base = neutral_start.as_ref().unwrap();
                        score -= self.continuation_log_probability(base, bytes, stop);
                    }
                    _ => {}
                }
                if score > best.0 {
                    best = (score, bytes.to_vec());
                }
            }
        }
        best.1
    }

    pub fn observe_bytes(&mut self, data: &[u8], learn: bool) {
        for &byte in data {
            for shift in (0..8).rev() {
                self.observe(((byte >> shift) & 1) as u32, learn);
            }
        }
    }

    /// Mean next-bit cross-entropy (bits/bit) over ``data``.
    pub fn score_bytes(&mut self, data: &[u8], learn: bool) -> (f64, u64) {
        let mut total = 0.0;
        let mut count = 0u64;
        for &byte in data {
            for shift in (0..8).rev() {
                let bit = ((byte >> shift) & 1) as u32;
                let p = self.predict();
                total -= if bit == 1 { p.log2() } else { (1.0 - p).log2() };
                count += 1;
                self.observe(bit, learn);
            }
        }
        (total, count)
    }

    /// Greedy decoding: append bytes until ``stop`` or ``max_bytes``.
    pub fn generate(&mut self, max_bytes: usize, stop: u8) -> Vec<u8> {
        let mut out = Vec::new();
        for _ in 0..max_bytes {
            let mut byte = 0u32;
            for _ in 0..8 {
                let p = self.predict();
                let bit = if p >= 0.5 { 1 } else { 0 };
                byte = (byte << 1) | bit;
                self.observe(bit, false);
            }
            if byte as u8 == stop {
                break;
            }
            out.push(byte as u8);
        }
        out
    }
}
