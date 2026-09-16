# R11 First Step: Glyph Vision (`reports/r11-glyph.json`, all gates pass)

Minimally interpreted 6x6 binary pixels (one channel per pixel, no
features) through the universal event path. Locked A -> B -> A mapping
switch (B inverts classes) with +-1px jitter and one noise pixel per
frame; 600-step phases (high-dimensional context detection needs longer
evidence than scalar streams).

6-seed tails: learn A 0.583 / B 0.600 / return 0.716 vs frozen 0.2,
shuffled <= 0.14, adapter-only persistence -0.6 on switched B. All 7
gates pass. First evidence the event protocol spans a spatial modality
with the unchanged core. Not claimed: real images, translation
invariance beyond +-1px, object semantics, or real-time deadlines (R12).
