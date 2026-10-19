# Paper Review Consistency Design

## Goal

Close the remaining consistency issues in reviewer items 3--5 without changing
the reconstruction protocol, model results, metric caches, or pending SSIM work.

## Design

- Treat the causal 3D KNN window length as `W=7`, comprising the current frame
  and six preceding same-UTC proxy fields. The Fig. 2 historical-field header
  will avoid reusing `W` and will state the historical range explicitly.
- Remove the obsolete comparison-region rectangle and its unused selection
  logic from the Fig. 4 generator. Regenerate the PNG and PDF and synchronize
  the paper copies.
- Keep the existing realized-fraction means and ranges, correct the LaTeX math
  formatting for `r`, and add the nominal-level by realized-bin counts. The
  row and column totals must remain 219 per nominal level and 145/237/275 per
  realized bin.

## Verification

- Run focused source/text consistency tests through pytest.
- Regenerate the affected figures and inspect the final raster outputs.
- Compile `_overleaf_review/main.tex` and check the log for LaTeX errors and
  undefined references.

