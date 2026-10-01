# R12 First Step: Embodied Cart (`research/reports/r12-cart.json`, all gates pass)

Deterministic 1-D cart (position/velocity, inertia, crash penalty) with an
independent fixed-gain brake controller, force clamp, track limits, and
telemetry. Goals vary per seed (goal-following, not fixed-point memory).

6-seed tails: learn 0.641 vs frozen 0.380, shuffled 0.251, passive 0.400;
zero emergency stops; engagements bounded. All 5 gates pass.

Recorded boundaries: closed-loop context switches under inertia plus
safety interventions defeat naive single-step credit (limit cycles, seed
luck); multi-step planning is defined next work. Safety engineering
(stronger soft bound) fixed both safety and learning: contained
exploration learns better.
