# Temporal Hourly V5 Equivalent Acceleration Design

## Status

Proposed and approved in principle. Implementation starts only after this design
is reviewed. The existing `temporal_hourly_v4` implementation remains unchanged.

## Goal

Reduce full JAXA hourly proxy generation time without changing the causal
information set, interpolation equations, source policy, or scientific output.
For the same input and configuration, V5 must preserve the V4 reconstruction
logic and show no measurable accuracy regression.

## Observed Bottleneck

V4 processes each annual series in chronological order, but it repeats two
expensive operations for every target time:

1. It reopens and rereads the same previous 72 raw frames, then rebuilds their
   KD-trees.
2. After batch KD-tree queries, it loops through every missing pixel in Python
   to concatenate candidates, select the top weighted candidates, calculate
   evidence, and mix proxy estimates.

The first full V4 run used only about one to one-and-a-half CPU cores per annual
worker despite `query_workers=21`; the per-pixel candidate calculations dominate
the small KD-tree query batches.

## Non-Negotiable Invariants

- Process each series on one strictly increasing real-timestamp timeline.
- Do not use another series, a future timestamp, or a nonexistent hour.
- Retain exactly the prior 1--72 real hours as raw/proxy history, skipping real
  timestamp gaps.
- Preserve raw-only temporal filling, `sigma=1.5` temporal-only Gaussian
  smoothing, source codes, confidence policy, proxy threshold, nonzero proxy
  distance rule, local-anchor bias rule, `k=20`, `source_k=8`, and the
  `rebuild_interval=500` progressive update boundary.
- Preserve the V4 candidate ordering: current candidates, then historical raw
  candidates in increasing lag order; historical proxy candidates in increasing
  lag order.
- V5 must not alter `temporal_hourly_v4`, v3, old H5 products, checkpoints, or
  paper caches.

## V5 Architecture

Create a separate package, `Data_Imputation/preprocessing/temporal_hourly_v5/`,
with the same input/output H5 contract as V4.

### Rolling Raw and Tree Cache

Each series keeps a rolling raw-frame cache alongside its existing proxy-frame
cache. On processing timestamp `t`, the pipeline obtains raw history only from
cached frames whose actual deltas are in `[1, 72]`; it then adds the current raw
frame after reconstruction. Missing timestamps are absent from the cache and
therefore skipped naturally.

Build each raw-frame KD-tree once at insertion and retain it with that raw frame
until it leaves the 72-hour cache. Build each proxy KD-tree only after its
confidence-filtered proxy frame has been accepted into the cache. The trees hold
the identical coordinates, values, confidence, and time lag that V4 would
construct at each target frame; they are cached computation, not new evidence.

### Batch-Vectorized Candidate Evaluation

For the existing deterministic batch of at most 500 residual pixels:

1. Query the unchanged current-frame, raw-history, and proxy-history trees.
2. Assemble the fixed-width candidate tensors in V4 order.
3. Apply the same finite-value checks, proxy eligibility, candidate weight,
   top-`k` selection, weighted estimate, evidence quality, local offset, blend
   alpha, and clipped confidence independently along each batch row.
4. Write the completed batch only after all rows use the single current-frame
   tree created at the beginning of that batch.

The final rule preserves V4 progressive semantics: values created in one batch
can enter the current tree for the next batch, but never influence another row
inside the current batch.

## Excluded Optimizations

Do not use approximate nearest-neighbor libraries, a single time-scaled 3D tree,
GPU approximate KNN, or changed candidate-pool limits. Those options can alter
which historical values are eligible and are excluded because output accuracy
must not decline.

## Validation Gates

1. Unit equivalence tests run V4 and V5 on controlled cases covering raw-only
   history, true time gaps, proxy blending, zero-distance proxy exclusion,
   sparse anchors, and progressive batches. Source code and fill masks must be
   exactly equal; SST and confidence must satisfy `atol=1e-6`, `rtol=0`.
2. A real chronological Series 8 smoke segment is run into a new temporary
   output root. H5 fields, timestamps, land NaNs, unresolved-ocean counts, and
   provenance masks must match V4's reference segment. SST differences must
   remain at or below `1e-6 degC`.
3. On the same artificially withheld raw-ocean pixels, MAE, Gradient MAE,
   gradient correlation, and SSIM must match within numerical rounding; no
   metric may decline.
4. Benchmark at least ten consecutive real frames using the same 21 KD-tree
   workers. V5 must demonstrate material wall-clock improvement before it is
   used for full production data generation.

## Output and Rollback

The interrupted V4 run left incomplete H5 files under
`/data/sst_data/sst_missing_value_imputation/jaxa_processed_data/`. They are not
valid final data and must remain untouched until V5 passes all validation gates.
After validation and before the V5 full run, remove only these newly generated
partial `h00`--`h23` directories and any V4 diagnostics in that exact root, then
write V5's full 9-series output to the user-specified path.

If any equivalence or metric gate fails, do not remove the partial V4 output or
start V5 production. Keep V4 untouched and report the failed gate.
