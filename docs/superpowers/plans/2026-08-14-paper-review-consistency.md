# Paper Review Consistency Implementation Plan

> **For agentic workers:** Execute these steps inline in order. No subagent is
> required because all changes share the same paper and figure artifacts.

**Goal:** Make reviewer items 3--5 internally consistent across paper text,
figure-generation code, and final figure assets.

**Architecture:** Use a deterministic bitmap-label postprocessor for the Fig. 2
asset whose original composition source is unavailable, simplify the existing
Fig. 4 generator, and update the relevant protocol paragraph in LaTeX. Focused
tests assert the stable text and source invariants.

**Tech Stack:** Python, Pillow, Matplotlib, pytest, LaTeX.

---

### Task 1: Add failing consistency checks

**Files:**
- Create: `Data_Imputation/tests/test_paper_review_consistency.py`

- [ ] Assert that the Fig. 2 label postprocessor exposes the approved label.
- [ ] Assert that the Fig. 4 generator has no zoom-box selection or drawing.
- [ ] Assert that the extra-mask paragraph contains the corrected math and the
      three nominal-by-realized-bin count vectors.
- [ ] Run the focused test and confirm that it fails against the current state.

### Task 2: Update Fig. 2

**Files:**
- Create: `Data_Imputation/visualization/paper_figs/update_fig2_proxy_label.py`
- Modify: `_overleaf_review/figure/result/fig2_causal_sst_proxy_construction.png`

- [ ] Implement an idempotent white-background title replacement using Pillow.
- [ ] Render `Historical proxy fields (t-6 to t-1)` in the existing title area.
- [ ] Run the script and inspect the complete figure.

### Task 3: Update Fig. 4

**Files:**
- Modify: `Data_Imputation/visualization/paper_figs/fig4_reconstruction.py`
- Modify: `Data_Imputation/visualization/paper_figs/output/fig4_reconstruction.png`
- Modify: `Data_Imputation/visualization/paper_figs/output/fig4_reconstruction.pdf`
- Modify: `_overleaf_review/figure/result/fig4_reconstruction.png`
- Modify: `_overleaf_review/figure/result/fig4_reconstruction.pdf`

- [ ] Remove the rectangle import, zoom selector, and mid-row drawing block.
- [ ] Use the title `Reconstruction Comparison at Three Nominal Extra-mask Levels`.
- [ ] Regenerate and synchronize both output formats.

### Task 4: Update realized-fraction reporting

**Files:**
- Modify: `_overleaf_review/main.tex`

- [ ] Change `mean (range) of (r)` to `mean (range) of \(r\)`.
- [ ] Add count vectors `143/60/16`, `2/174/43`, and `0/3/216` for nominal
      levels 30%, 55%, and 75%, respectively.
- [ ] State that pooling yields the existing `145/237/275` figure totals.

### Task 5: Verify final artifacts

- [ ] Run `pytest -q Data_Imputation/tests/test_paper_review_consistency.py`.
- [ ] Inspect both regenerated PNG figures at original resolution.
- [ ] Compile the paper and scan the log for errors and undefined references.

