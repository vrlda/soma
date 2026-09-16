//! Differential learning-kernel runner: fixture in, updated strengths out.
//!
//! Applies the Hebbian then actor update exactly once, mirroring a single
//! rewarded Python step with pre-seeded traces. Routing-coupled module
//! updates stay Python-side by design.

use soma_engine::neuron::{ForwardFixture, Network};
use std::io::Read;

#[derive(serde::Deserialize)]
struct LearnFixture {
    #[serde(flatten)]
    base: ForwardFixture,
    learning_rate: f64,
    actor_learning_rate: f64,
    prediction_error: f64,
}

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin");
    let fixture: LearnFixture = serde_json::from_str(&input).expect("fixture");
    let mut network = Network::from_fixture(&fixture.base);
    network.apply_reward(fixture.learning_rate, fixture.prediction_error);
    network.apply_actor_reward(fixture.actor_learning_rate, fixture.prediction_error);
    let strengths = network.strengths();
    let mut out = String::from("{\"strengths\":[");
    for (index, (source, destination, strength)) in strengths.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format!("[{:?},{:?},{:?}]", source, destination, strength));
    }
    out.push_str("]}");
    println!("{}", out);
}
