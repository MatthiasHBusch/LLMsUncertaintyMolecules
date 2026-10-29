# Every claim of the manuscript, and what stands behind it

One row per assertion of `paper/manuscript.tex` and `paper/appendix.tex` (restructured version of
2026-10-14, three questions Q1 / Q2 / Q3). `#` is the claim id printed by `src/verify_paper.py`;
`basis` is the file the number comes from; `status` is the outcome of the run of 2026-10-14
(108 OK, 5 CODE, 7 DOC-ONLY, 2 INFO, 0 DIFF); after the ESOL unit rule of 2026-10-15 (label-sign plausibility rule; rows 2.1.10 and D.5
new / now checked, rows 2.5.6, 3.4.2–3.4.5, 3.4.12, D.6, E.6 updated): 110 OK, 5 CODE, 6 DOC-ONLY, 2 INFO, 0 DIFF. Layout revision of 2026-10-14: Table 2 and Tables A4
(contamination per cell), A9–A11 (ladder tokens, ladder deltas, operations) removed, their per-cell values
stay in the repository; rows G.4 and H.7, which checked those tables only, removed; Fig. 5
(contamination) added with rows 3.4.11–3.4.14 and D.6.
Review round of 2026-10-28: the calibration analysis left the paper (Fig. fig4_calibration, Tables
c_e1_calibration / c_e2_uc3; `calibration_cells.csv` stays in the repository); rows 2.5.4, 2.5.5, 3.3.6–3.3.8,
D.2 and the old E.1–E.5 removed; Appendix E is now "Repeat spread and similarity to the context": the old E.6
is E.1, new rows E.2 (mixture σ, `src/mixture_sigma.py`), E.3 and E.4 (similarity to the context,
`src/similarity_context.py`); C.2 (ensembles), D.1 and 3.4.5 reworded to the RMS of e/σ; Abs.5 maps to
3.4.1, D.1, 3.4.5. Run of 2026-10-28: 103 OK, 4 CODE, 6 DOC-ONLY, 2 INFO, 0 DIFF.

```bash
python src/verify_paper.py            # all sections, manuscript order
python src/verify_paper.py --sec 3.2  # one subsection (2.1 ... 2.6, 3.1 ... 3.4, 4, C ... H)
```

Status: ✅ recomputed from `results/`, `data/splits` or `src/registry` and equal to the printed value;
✅ code = a constant read from the source file; DOC-ONLY = stated only in a design document or
established by running a separate script (check by hand); INFO = printed for orientation, not compared.

Conventions (Sections 2.3–2.5): language models at effort `low`; values are context draw 1; `[a–b]` =
min–max over the three context draws at N ≤ 120; baselines averaged over the seeds of a draw, except
the paired bootstrap (seed 0); the ensembles have seed-0 files only (C.2), and the ESOL recall split
(3.4.3, 3.4.4, Table D2) reads `calibration_cells.csv`, whose TabPFN rows are seed 0.

---

## 2.1 Datasets and contamination control

| # | claim | basis | status |
|---|---|---|---|
| 2.1.1 | 4,200 / 1,513 / 1,006 / 1,128 molecules | `data/splits/<ds>_v2.json` `n_total` | ✅ |
| 2.1.2 | corpus filter excluded 20 of the first 170 lipophilicity candidates, none on BACE / antiviral (Appendix B) | splits `excluded_present`, `positions_checked` | ✅ |
| 2.1.3 | test sets of 150 molecules | splits `test_ids` | ✅ |
| 2.1.4 | training pools 4,042 / 1,355 / 844 / 976 (Table 1a, Table A2) | splits `n_train_pool` | ✅ |
| 2.1.4b | 8 / 8 / 12 / 2 pool molecules removed by the InChIKey-14 rule (Table A2) | splits `n_removed_dedup` | ✅ |
| 2.1.4c | seven open corpora searched | `corpus/summary_by_subset.csv` `index` | ✅ |
| 2.1.5 | none of the 450 clean test molecules in any corpus in any form | same, subset `v2test150` | ✅ |
| 2.1.6 | a third of the ESOL test molecules occur as SMILES in Dolma (Appendix D) | same, `pct_present_smiles` 32.4 | ✅ |
| 2.1.7 | 21 of 21 clean-dataset cells clean; ESOL flagged for 6 of 7 (Fig. 5, Table 1) | `zeroshot/cells.csv` `verdict` | ✅ |
| 2.1.8 | 81 of 150 lipophilicity labels with three significant figures | same, `n_usable` | ✅ |
| 2.1.9 | screen dated 2026-10-09, GPT-6 Astra 2026-10-13 | `zeroshot/*.json` `meta.timestamp` | ✅ |
| 2.1.10 | ESOL unit rule: positive zero-shot answers to molecules with a negative label read as mol/L and converted to logS (answers to the 10 positive-label molecules kept); 41 (Sonnet) and 30 (Haiku) converted, 0–5 for the other models, none on the clean datasets | `zeroshot/cells.csv` `n_unit_converted` (= positive raw answers on negative labels in `zeroshot/*.json`); `src/score_zeroshot.py` `UNIT_RULES` | ✅ |

## 2.2 In-context protocol and model panel

| # | claim | basis | status |
|---|---|---|---|
| 2.2.1 | labels with two decimals, 68 % definition, JSON keys (Appendix A) | `src/icl_common.py` | DOC-ONLY |
| 2.2.2 | σ floored at 0.01 for scoring | `src/analyze_icl.py` `SIGMA_FLOOR` | ✅ code |
| 2.2.4 | seven panel models | `results/icl/cells.csv` | ✅ |
| 2.2.5 | pinned provider, fallback disabled | `src/registry/models_v2.json` | ✅ |
| 2.2.6 | flex tier for the OpenAI and Google models | registry `service_tier` | ✅ |
| 2.2.7 | completion cap 32,000 | `src/julia/Common.jl` `max_tokens` | ✅ code |
| 2.2.8 | temperature and top-p not sent (provider default) | `docs/EXPERIMENT_DESIGN_v2.md` §4 | DOC-ONLY |
| 2.2.9 | Opus 5.5 answered at N = 30, content filter in every call at 250 and 1,000 (also Appendix B) | `results/meta/(diagnostics/)icl_probe_2026-10-08*.jsonl` | ✅ (2 / 2 / 4 calls) |

## 2.3 Context sizes, repeats, draws

| # | claim | basis | status |
|---|---|---|---|
| 2.3.1 | N = 30, 60, 120, 250, 500, largest 1,000 / 1,000 / 844 / 976 | `cells.csv` `size` | ✅ |
| 2.3.2 | R = 3 calls per molecule | `cells.csv` `n_calls / n_mol` | INFO (median 3.00) |
| 2.3.3 | first 120 molecules of two draws share at most 20 (expected 17.1) | `results/meta/context_draw_overlap.csv` | ✅ |
| 2.3.4 | 168 main-run cells, 168 draw cells (Appendix E) | `cells.csv` | ✅ |
| 2.3.5 | every main-run cell 150 molecules, parse rate ≥ 0.995 (Appendix A), no truncation | `cells.csv` | ✅ (min parse 0.9956) |
| 2.3.5b | context draws: parse rate ≥ 0.9978, two cells with 149 molecules | `cells.csv` | INFO |

## 2.4 Classical baselines

| # | claim | basis | status |
|---|---|---|---|
| 2.4.1 | eight methods | `results/baselines/summary.csv` | ✅ |
| 2.4.2 | TabPFN 159–201 descriptors per cell (Table A3) | `tabpfn__*.meta.json` `n_features` | ✅ |
| 2.4.2b | TabPFN σ = (q84 − q16)/2 | same, `sigma_definition` | ✅ |
| 2.4.3 | RF 200 trees; Morgan radius 2, 2,048 bits (Table A3) | `rf*__*.meta.json` | ✅ |
| 2.4.4 | GP ML-II from 14 starts (Appendix C) | `gp_tanimoto__*.meta.json` | ✅ |
| 2.4.5 | kNN k = 5 (Table A3) | `knn_tanimoto__*.meta.json` | ✅ |
| 2.4.6 | five-member Chemprop and MLP ensembles | `*_ens__*.meta.json` | ✅ |
| 2.4.7 | seeds 0–2 for TabPFN, both RF, GP, kNN in draw 1; GP, kNN in draws 2–3 (Table A3) | `summary.csv` `seed` | ✅ |
| 2.4.9 | tabpfn 9.1.0, chemprop 2.0.0 (Appendix C, H, data availability) | `*.meta.json` | ✅ |
| 2.4.10 | TabPFN v3.5 trained on synthetic tasks only | vendor model card, DES §7 | DOC-ONLY |
| 2.4.11 | leakage self-check bit-identical (Appendix C) | `src/baselines/selfcheck.py` (run it) | DOC-ONLY |

## 2.5 Metrics and statistics

| # | claim | basis | status |
|---|---|---|---|
| 2.5.1 | ties after rounding σ to 10 decimals | `src/selective_prediction.py` `TIE_DECIMALS` | ✅ code |
| 2.5.3 | median 30 distinct σ per main-run cell | `selective_prediction.csv` `n_sigma_distinct` | ✅ |
| 2.5.6 | ESOL 63 recalled / 50 never / 37 unscorable (Section 3.4, Appendix D) | `cells.csv` | ✅ |
| 2.6.1 | 1,000 resamples, `default_rng(0)` | `src/bootstrap_cells.py` `B, BOOT_SEED` | ✅ code |
| 2.6.2 | differences against TabPFN and the GP | `bootstrap_cells.csv` | ✅ |
| 2.6.3 | per draw and combined at N ≤ 120, draw 1 at the largest N | `bootstrap_cells.csv` `draw` | ✅ |

## 2.6 Reasoning ladder

| # | claim | basis | status |
|---|---|---|---|
| 2.7.1 | six ladder models, no GPT-6 Astra | `ladder_cells.csv` | ✅ |
| 2.7.2 | effort none rejected by four of the six ladder models | `ladder_cells.csv` | ✅ |
| 2.7.3 | 160 ladder cells beyond the main run (Appendix G) | `cells.csv` `reasoning != low` | ✅ |
| 2.7.4 | an Astra ladder would have cost at least the main run of all seven models (Appendix G: 4.8 / 6.0 ×, USD 315–393 vs 311.02) | `cells.csv` `cost_usd` | ✅ |
| 3.1.12 | Gemini high: 15–35 % of the calls truncated, 126–150 molecules kept (Sections 2.6, 3.1; Fig. 2) | `ladder_cells.csv` | ✅ |

---

## 3.1 Q1: accuracy over context size and reasoning effort

| # | claim | basis | status |
|---|---|---|---|
| 3.1.1 | RMSE falls by 8 to 37 % from N = 30 to the largest N on the clean sets | `cells.csv` | ✅ (7.6 / 36.5 %) |
| 3.1.2 | antiviral N = 60 above N = 30 for 7 / 0 / 0 models in draws 1 / 2 / 3 | `cells.csv` | ✅ |
| 3.1.3 | Sonnet 5.5 below every baseline at N = 30, every dataset and draw | `cells.csv`, `summary.csv` | ✅ 12 of 12 |
| 3.1.4 | lipophilicity 0.60 [0.60–0.60] vs TabPFN 1.17 [1.05–1.17] | same | ✅ |
| 3.1.5 | combined N = 30 RMSE diff to TabPFN excludes 0 for 6 / 3 / 1; antiviral only Sonnet | `bootstrap_cells.csv` | ✅ |
| 3.1.6 | largest N lipophilicity −0.14 [−0.21, −0.07] | `calibration_bootstrap.csv` (= `bootstrap_cells.csv`) | ✅ |
| 3.1.7 | largest N BACE not distinguishable | same | ✅ |
| 3.1.8 | antiviral: RMSE diff excludes 0, MAE diff does not | same | ✅ |
| 3.1.9 | largest N: Sonnet and Astra among the three most accurate on each clean set; Luna / GLM less accurate than TabPFN in most BACE + antiviral cells (22 of 24) and from N = 500 on lipophilicity | `cells.csv`, `summary.csv` | ✅ |
| 3.1.10 | Sonnet 0 tokens at every level; Haiku from medium; Gemini up to 30,717 | `ladder_cells.csv` | ✅ |
| 3.1.11 | RMSE high − low between −0.12 and +0.06 over 36 clean cells, both signs; Luna ESOL 0.56 → 0.81 at effort none | same | ✅ |
| 3.1.13 | ESOL flat: Sonnet 0.30 → 0.29 while TabPFN 0.97 → 0.51 | `cells.csv`, `summary.csv` | ✅ |
| 3.1.14 | lipophilicity: LLMs improve by 18–37 % from N = 30 to 1000, TabPFN and Chemprop by 46–48 % | `selective_prediction.csv` RMSE | ✅ |

## 3.2 Q2: which uncertainty flags the molecules a model gets wrong

| # | claim | basis | status |
|---|---|---|---|
| 3.2.1 | median ρ 0.31 over 126 clean LLM cells vs 0.36 over 18 TabPFN cells | `selective_prediction.csv` `rho` | ✅ |
| 3.2.2 | no LLM cell on the clean datasets above 0.46 | same | ✅ (BACE, Astra, N = 120) |
| 3.2.3 | lipophilicity N = 30: Sonnet 0.36 [0.36–0.40] vs TabPFN 0.13 [0.10–0.15]; combined diff excludes 0 for Sonnet and Astra; 5 of 7 above every baseline in draw 1 | same; `bootstrap_cells.csv` | ✅ |
| 3.2.4 | BACE N = 30: no LLM interval excludes 0 against TabPFN or the GP; antiviral: TabPFN better than 4 of 7 | `bootstrap_cells.csv` | ✅ |
| 3.2.5 | TabPFN lipophilicity ρ 0.13 → 0.38; LLM median ρ does not rise with N on any clean set | `selective_prediction.csv` | ✅ |
| 3.2.6 | N ≥ 250: TabPFN ρ 0.37–0.44 on lipophilicity and antiviral vs LLM medians 0.26–0.38; BACE LLM medians 0.31–0.34 vs TabPFN 0.25–0.27 | same | ✅ |
| 3.2.7 | largest N: ρ diff to TabPFN excludes 0 only for Haiku on lipophilicity (in TabPFN's favour) | `bootstrap_cells.csv` | ✅ |
| 3.2.8 | UC3 zero for up to a third of a cell; ranks better than UC1 in 8 of 126 clean cells | `calibration_cells.csv` | ✅ |
| 3.2.9 | ρ high − low median +0.02 over 36 clean ladder cells | `ladder_cells.csv` + `selective_prediction.csv` | ✅ |

## 3.3 Q3: using the most certain predictions without measuring

| # | claim | basis | status |
|---|---|---|---|
| 3.3.1 | N = 30 draw 1, best LLM minus best baseline RMSE50: lipophilicity −0.71, BACE −0.01, antiviral −0.06, so clearly ahead only on lipophilicity; below every baseline in 9 of 9 dataset × draw; lipophilicity Sonnet 0.44 [0.42–0.44] vs TabPFN 1.15 [0.99–1.15] | `selective_prediction.csv` | ✅ |
| 3.3.2 | combined gain50 diff excludes 0 only on lipophilicity (Sonnet, Haiku, Gemini); Sonnet / Haiku 0.27 / 0.31 vs TabPFN 0.02; BACE and antiviral within 0.06 | `bootstrap_cells.csv`, `selective_prediction.csv` | ✅ |
| 3.3.2b | BACE draw 1 0.77 vs 0.78; excludes 0 only with the draws combined | same | ✅ |
| 3.3.7 | antiviral N ≥ 60: TabPFN gain50 0.33–0.51 vs LLM medians 0.16–0.32; lipophilicity N ≤ 120: LLM median gain50 ≥ baseline median | `selective_prediction.csv` | ✅ |
| 3.3.3 | antiviral N = 120 combined: TabPFN RMSE50 lower than every LLM, excludes 0 for 4 of 7; largest N 0.31 vs 0.32, CI includes 0 | `bootstrap_cells.csv`, `selective_prediction.csv` | ✅ |
| 3.3.4 | lipophilicity and BACE: best LLM RMSE50 below TabPFN at every N; largest N Astra 0.34 vs 0.43 (excludes 0) and 0.46 vs 0.61 (includes 0) | same | ✅ |
| 3.3.5 | q = 0.2 antiviral N = 120: TabPFN 0.27 vs LLM 0.43–0.54; lipophilicity and BACE best LLM ≤ TabPFN at every N | `src/make_figs_paper.py::_rmse_q_rows(0.2)` on the per-molecule files | ✅ |

## 3.4 Contamination

| # | claim | basis | status |
|---|---|---|---|
| 3.4.1 | ESOL: no LLM lowers its RMSE by more than 0.08 between N = 30 and 976 | `cells.csv` | ✅ |
| 3.4.2 | 63 recalled / 50 never recalled | `cells.csv` | ✅ |
| 3.4.3 | factor 1.9–2.4 for the four highest-hit3 models at N = 976 | `calibration_cells.csv`, `zeroshot/cells.csv` | ✅ |
| 3.4.4 | TabPFN and GLM 5.3 Flash (not flagged) factor 1.3 | same | ✅ |
| 3.4.5 | Sonnet 5.5 at N = 976: RMS of e/σ (σ floored at 0.01) 0.42 recalled / 1.27 never recalled | `esol_permolecule.csv` `uc1`, draw 1 (= `calibration_cells.csv` `s`) | ✅ |
| 3.4.11 | zero-shot Spearman ρ with the labels on the clean datasets 0.09 to 0.83 (Fig. 5a) | `zeroshot/cells.csv` `spearman` | ✅ (GLM BACE / Sonnet lipophilicity) |
| 3.4.12 | ESOL: all seven models reach a zero-shot ρ of 0.91 to 0.98 (Fig. 5a) | same | ✅ |
| 3.4.13 | Fig. 5 caption: 81 / 143 / 136 / 113 labels with three significant figures scored | `zeroshot/cells.csv` `n_usable` | ✅ |
| 3.4.14 | Fig. 5 caption: 136 ESOL test molecules with a searchable string | `corpus/summary_by_subset.csv` `n_mol` | ✅ |
| 3.4.15 | ESOL LLM median ρ 0.45 vs 0.31 on the clean datasets (conclusion) | `selective_prediction.csv` | ✅ |

## 4 Discussion

| # | claim | basis | status |
|---|---|---|---|
| 4.1 | cost of a cell at the largest N differs by two orders of magnitude between panel models (USD 0.13 to 16.48) | `cells.csv` `cost_usd` | ✅ |
| 4.2 | bootstrap interval of a single ρ about ±0.15 (Limitations, Appendix F) | `bootstrap_cells.csv` | ✅ |
| 4.3 | which model when: synthesis of 3.1–3.3, no number of its own | rows 3.1.3, 3.1.9, 3.2.3–3.2.7, 3.3.1–3.3.4 | — |
| 4.4 | reasoning: synthesis of 3.1.10–3.1.12 and Appendix G | rows 3.1.10, 3.1.11, 3.1.12, G.2 | — |
| 4.5 | contamination / ESOL: synthesis of 3.4 and Appendix D, E | rows 3.4.1–3.4.5, D.1, E.1 | — |
| 4.6 | limitations: 150 molecules, three clean sets, draws only at N ≤ 120, one prompt, one zero-shot call per model, corpus search a proxy, ChEMBL origin, Gemini censored, no Astra ladder, dated endpoints | rows 2.1.3, 2.1.8, 2.3, 2.6, 3.1.12, 2.7.1, H.5 | — |

---

## Appendix

| # | claim | basis | status |
|---|---|---|---|
| C.1 | GP ML-II at N = 30 on a bound in all 12 cells: 8 lower noise, 4 lower amplitude | `gp_tanimoto__*__N30*.meta.json` (seed 0, draws 1–3) | ✅ |
| C.2 | RMS of e / member spread (floored at 0.01) over the draw-1 cells: 3.8–26.8 (Chemprop ensemble), 8.9–31.8 (MLP ensemble) | `results/baselines/<chemprop_ens\|mlp_ens>__<ds>__N<size>.csv`, draw 1, seed 0 | ✅ |
| C.3 | TabPFN fit 9 to 17 s at N = 30, at most 98 s | `tabpfn__*.meta.json` `wall_s` | ✅ |
| C.4 | 217 RDKit 2D descriptors, per-cell filter, median imputation | `docs/EXPERIMENT_DESIGN_v2.md` §7 | DOC-ONLY |
| D.0 | 7,753 of 7,847 molecules searchable | `corpus/summary_by_subset.csv` `n_mol` | ✅ |
| D.1 | ESOL: RMSE recalled < never recalled in every cell (54 suite cells, 84 LLM cells over the draws); mean stated σ recalled < never in 42 of 42 LLM cells; RMS of e/σ recalled < never in 42 of 42 LLM cells | `calibration_cells.csv`, `cells.csv`, `esol_permolecule.csv` | ✅ |
| D.3 | GLM ESOL RMSE 0.74 → 0.73 | `cells.csv` | ✅ |
| D.4 | zero-shot screen: all replies parsed, none truncated | `zeroshot/cells.csv` | ✅ |
| D.5 | unit rule vs the verbatim 45 scoring: Sonnet hit3 31.0 → 37.2 %, Haiku 8.8 → 10.6 % (partial → heavy), ρ 0.27 / 0.51 → 0.94 / 0.94; no match lost; no molecule recalled → never, one (Mecarbam) never → recalled | `zeroshot/_pre_unitfix_2026-10-15/` vs `zeroshot/` `cells.csv`, `esol_v2test150_permolecule.csv` | ✅ |
| D.6 | ESOL regimes: heavy for Sonnet, GPT-6.1 Sol, GPT-6 Astra, Gemini, Haiku; partial for GPT-6 Luna (also Table 1b) | `zeroshot/cells.csv` `regime` | ✅ |
| E.1 | repeat spread (UC3) exactly zero for 0.7–33 % of a cell; ranks better than UC1 in 8 of 126 clean and 24 of 42 ESOL cells; Sonnet ESOL N = 976 0.65 vs 0.40; 38 % vs 18 % identical repeats | `calibration_cells.csv` `share_sigma_zero`, `rho_sigma_err` | ✅ |
| E.2 | mixture σ of the three repeats: median ρ_mix − ρ_UC1 0.00 on the 126 clean cells (higher in 68), +0.03 on ESOL (higher in 42 of 42) | `mixture_sigma.csv` draw 1 (`src/mixture_sigma.py`) | ✅ |
| E.3 | ρ(largest Tanimoto similarity to the context, \|e\|) on the clean datasets, draw 1: LLM median −0.21 at N = 30 and −0.12 at the largest N, LLM range −0.31 to 0.01 over both sizes; four display baselines −0.37 to 0.15 (−0.37 to 0.15 at N = 30, −0.36 to 0.00 at the largest N; LLMs −0.31 to 0.01 and −0.28 to 0.00) | `context_similarity.csv` `rho_sim_abserr` draw 1 (`src/similarity_context.py`) | ✅ |
| E.4 | ρ(similarity, σ) on the clean datasets, medians at N = 30 / largest N over the 21 LLM cells (7 models × 3 datasets) and over the 3 datasets for a baseline: LLMs −0.56 / −0.49; TabPFN −0.54 / −0.28; GP −0.97 / −0.89 | `context_similarity.csv` `rho_sim_sigma` draw 1 | ✅ |
| F.1 | q = 0.8 orders the models as the RMSE does (rank correlation ≥ 0.69 over the 18 clean cells) | `selective_prediction.csv` | ✅ |
| F.2 | Fig. A5: most accurate LLMs left of every baseline on 3 of 3 clean sets, above them only on lipophilicity | same | ✅ |
| F.3 | the cut splits a tied group in 286 of 336 cells; tie range median 0.03, at most 0.23 in a draw-1 cell | same | ✅ |
| F.4 | draw SD of the RMSE lower for the LLMs in 10 of 12 dataset × N | `report_extras.json` `draws` | ✅ |
| F.5 | gain50 range over draws at N = 30 up to 0.22 | `selective_prediction.csv` | ✅ |
| G.1 | outside Gemini high at most 2 of 450 calls truncated | `ladder_cells.csv` | ✅ |
| G.2 | Gemini reports no reasoning tokens in 18 % of its main-run calls | `report_extras.json` | ✅ |
| G.3 | ladder steps cost USD 3.01 / 71.62 / 78.65 / 241.68; Gemini high 171.70 | `cells.csv` `cost_usd` | ✅ |
| G.5 | five of the seven panel endpoints reject the effort none (Table 1b) | `effort_probe_*.jsonl`, `ladder_cells.csv` | ✅ |
| H.1 | cells with fewer than 450 calls are all OpenAI cells, at least 445 calls, every molecule answered | `cells.csv` | ✅ |
| H.2 | cache share 0.91–1.00 at N ≥ 120 | `cells.csv` `cache_share` | ✅ |
| H.3 | 7 to 37 distinct σ among the 450 replies of a cell | `report_extras.json` `sigma_distinct` | ✅ |
| H.4 | 1,613 duplicated records in the context draws | raw `*__d[23].jsonl` vs `cells.csv` | ✅ |
| H.5 | main run 2026-10-09..13, ladder 2026-10-12..13, draws 2026-10-13 | `results/icl/*.jsonl` `timestamp` | ✅ |
| H.6 | prices and endpoint lists fetched 2026-10-08 | registry `prices_fetched` | DOC-ONLY |

---

## Abstract, Introduction, Conclusion

These sections introduce no number of their own: every number is one already stated and checked above,
and every claim is a restatement of a row. `verify_paper.py` does not cover them.

| # | section | claim | maps to |
|---|---|---|---|
| Abs.1 | abstract | 7 models, 4 datasets, three corpus-absent and zero-shot clean, ESOL control, classical baselines on the same molecules | 2.2.4, 2.1.1, 2.1.5, 2.1.7, 2.4.1 |
| Abs.2 | abstract | every model improves with the context; at 30 examples the most accurate beats every baseline; reasoning effort changes accuracy little | 3.1.1, 3.1.3, 3.1.11 |
| Abs.3 | abstract | stated σ ranks errors weakly (ρ up to 0.46), beats a classical σ only on lipophilicity at small contexts; TabPFN as good or better on two of three clean sets at larger contexts | 3.2.2, 3.2.3, 3.2.6 |
| Abs.4 | abstract | certain half of the best LLM clearly the most accurate at small contexts only on lipophilicity; on BACE and antiviral a classical baseline on par from 30 | 3.3.1, 3.3.2, 3.3.3 |
| Abs.5 | abstract | ESOL: no improvement with the context; uncertainty smaller on recalled molecules but by less than the error | 3.4.1, D.1, 3.4.5 |
| 1.1 | intro | scope paragraph (7 models, 4 datasets, 150 test molecules, N = 30–1,000, R = 3, three draws, eight baselines, six ladder models) | 2.1.1–2.1.5, 2.2.4, 2.3.1, 2.4.1, 2.7.1 |
| 1.2 | intro | the two uses named separately: measurement selection (ρ; the error of the certain molecules is not the criterion) and selective prediction (error of the kept fraction) | definitional, no number |
| 1.3 | intro | cited-literature statements (Vacareanu, Requeima, Xiong, Mirza, Busch2026Memorization / Blinding) | DOC-ONLY: read against the cited abstracts |
| 1.C | intro | Contributions bullets 1–4 | 2.1.5, 2.1.7, 2.4.1, 3.1.1, 3.1.3, 3.1.11, 3.2.3, 3.2.6, 3.3.1, 3.3.3, 3.4.1, 3.4.5 |
| 5 | conclusion | restates Abs.2–Abs.5; one number (antiviral, TabPFN's certain half from 120 molecules) | 3.3.3 |
