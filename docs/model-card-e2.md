# Model Card: E2 English Sequence Memory (Demonstration, Not Released)

- Architecture: order-16 bounded suffix circuits (131,072 ceiling) + episodic
  rules + turn-scoped dialogue tables + uncertainty-gated motor fusion.
- Data: 14 public-domain Gutenberg books, 11.5 MB acquisition, 141 KB
  validation (Jekyll), 181 KB sealed test (Time Machine). Manifest:
  `reports/e2-manifest.json`.
- Metrics: validation 0.3979, sealed test 0.3995 bits/bit vs 0.559 bar;
  fused E0 parity 0.37509; probe fallback documented in `reports/`.
- Reproduce: `python3 r3b_e1_benchmark.py --manifest reports/e2-manifest.json`.
- Intended use: research baseline for continual bit-level language
  experiments. Not a chat product: no comprehension, fluent nonsense
  answered from marginals, 766 MB–1.8 GB RSS (needs R5 engine).
- Limitations: byte-identity via memorization+backoff (no semantic
  transfer at scale), stochastic grammar unacquired, multi-output
  control unproven, no GPU path.
- Safety: quarantine for untrusted ingestion, trust-ranked corrections,
  exact forgetting, red-team gates in `reports/r9-redteam.json`.
