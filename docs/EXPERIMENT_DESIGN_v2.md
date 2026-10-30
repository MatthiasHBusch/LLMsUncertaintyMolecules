# v2 experiments -- design as decided 2026-09-08 (panel changed 2026-10-08, final panel and ICL prompt revision 2026-10-09; probe results sections 9 and 10)

Paid so far: the two probes of 2026-10-08 ($5.28, section 9), the opus55 size diagnostic of
2026-10-08 ($0.12) and the MiMo probes of 2026-10-09 ($0.06, section 10); the zero-shot check
(section 3) was started on 2026-10-09 and is running (`results/contamination/zeroshot/`). Every
runner has a dry-run flag and a hard budget.

**Panel decision 2026-10-09 (project owner; supersedes the two 2026-10-08 panels below).** Six
models: Claude Haiku 5.5 (`haiku55`), Claude Sonnet 5.5 (`sonnet55`), GPT-6 luna (`gpt6luna`,
flex), GPT-6.1 sol (`gpt61sol`, flex), Gemini 3.8 Flash (`gem38flash`, flex), GLM 5.3 Flash
(`glm53flash`). Dropped: Opus 5.5 (Anthropic's classifier refuses the ICL prompt from N >= 250),
DeepSeek V4.1 Flash (blocked by the OpenRouter account's privacy setting), Kimi K3 (too
expensive), MiMo V2.6 (reasoning either off or unbounded). Details: section 4 (table), section 10
(reasons, MiMo probe, cost). None of the six was in the 45 audit, so the zero-shot check is the
first contamination measurement for every panel model. Same day: the ICL prompt was revised
(section 4, "Prompt revision 2026-10-09"); temperature 1.0 and R = 1 first, then R = 3, confirmed.

**Panel change 2026-10-08 (decision of the project lead; superseded 2026-10-09).** The model panel is now seven models
(section 4): the Anthropic family across three sizes (Claude Haiku / Sonnet / Opus 5.5), so that
model size can be read within one family, plus breadth of providers -- OpenAI GPT-5.6 terra,
Google Gemini 3.8 Flash and two Chinese open-weight models (DeepSeek V4.1 Flash, Kimi K3).
Of the former panel (Opus 5, GPT-5.6 sol, Gemini 3.6 Flash, Grok 4.5, Kimi K3) Kimi K3 stays;
the other four are kept as reserve in the registry, together with Grok 4.7, GLM 5.3 and
Qwen3.8 2.4T. **Same evening (decision of the project lead): GPT-6.1 sol (`gpt61sol`, flex)
replaces GPT-5.6 terra in the panel**; terra stays as reserve, and eight more reserves were added
(GPT-6 luna, GPT-6 astra -- probe only, GLM 5.3 Flash, Mistral Large 4, Hy4 preview, Step 5
Preview, DeepSeek V4 Pro 0813, Qwen3.8 27B). **Decision record 2026-10-13 (project lead, after the
R = 3 results): GPT-6 Astra joins the panel as the seventh model** (registry `in_panel: true`), to add a
stronger OpenAI model next to Sonnet 5.5; zero-shot screen, main run R = 3 (all sizes, $149) and the
context draws were run for it on 2026-10-13. Astra is NOT part of the reasoning ladder (sections
13-14), which had been completed before: the ladder is therefore a declared six-model sub-experiment;
an Astra ladder (~$300 at the measured Astra cost per cell) is not planned unless the lead decides so. Registry: `src/registry/models_v2.json` (replaces `src/memo45/models_45.json` for
v2; endpoints, pinned providers without fallback, flex tier, prices incl. cache read, fetched
2026-10-08 from OpenRouter). The 45 contamination verdicts below were measured for the OLD panel;
of the new panel only Kimi K3 was in the 45 audit (terra, the other 45 model, has left the panel),
so the zero-shot check (section 3) has to establish the clean / contaminated status of the other
six before the main run.

## 1. Question

Does a frontier LLM used as a direct in-context regressor produce a usable self-reported
uncertainty, how does that change with the number of in-context training examples and with
the reasoning effort -- measured on model x dataset combinations where the model has NOT
memorised the labels, and contrasted with one dataset it has (ESOL)?

## 2. Datasets, splits, test sets (`src/make_splits.py`, `data/splits/<ds>_v2.json`)

| dataset | n | 45 verdict (former panel; new models pending the zero-shot check) | test | training pool | in-context ladder |
|---|---|---|---|---|---|
| lipophilicity | 4200 | clean | 150 | 4042 (4050 - 8 dedup) | 30 / 60 / 120 / 250 / 500 / 1000 |
| bace | 1513 | clean | 150 | 1355 (1363 - 8 dedup) | 30 / 60 / 120 / 250 / 500 / 1000 |
| antiviral | 1006 | clean (recency control) | 150 | 844 (856 - 12 dedup) | 30 / 60 / 120 / 250 / 500 / 844 |
| esol | 1128 | contaminated for all 5 panel models (Opus 5 62 %, Grok 4.5 13 %, sol 12 %, Kimi K3 8 %, Gemini 3.6 Flash 5 % verbatim) | 150 | 976 (978 - 2 dedup) | 30 / 60 / 120 / 250 / 500 / 976 |

- FreeSolv is discarded from everything (642 molecules cannot supply a separate 1000-sample pool).
- ESOL has 1128 molecules, 22 short of 150 + 1000; the top rung is the full pool (976 after
  dedup). Alternative if an exact 1000 rung matters: 128 test molecules.
- **Training-pool dedup (decision 2026-10-08, project lead).** The pools contained structures
  that also occur in the test sets (ESOL: test "Guaiacol" = pool "o-Methoxyphenol", test
  "Dialifor" = pool "Dialifos", same SMILES / same label). Rule (`make_splits.py`): after the test
  set is fixed, every pool molecule whose InChIKey connectivity block (first 14 characters of the
  RDKit InChIKey of the canonical SMILES) equals that of ANY test molecule is removed -- exact
  duplicates and stereoisomers / enantiomers of test molecules alike. Removed: lipophilicity 8
  (7 test molecules; all enantiomers or stereo-unspecified forms, labels differ by 0.01-0.35),
  BACE 8 (7 test molecules; 4 stereoisomers, 4 tautomer / protonation forms of which 2 carry the
  identical label), antiviral 12 (11 test molecules; all stereoisomers, no identical label,
  |delta pIC50| up to 3.1), ESOL 2 (both exact duplicates under another name, identical label). Pairs:
  `results/contamination/train_pool_dedup.csv`. Pool order otherwise unchanged (same seed and
  permutation, duplicates dropped), so N = 30 / 60 cells are identical to before everywhere and
  N = 120 changes only for lipophilicity (first removal at pool position 110; BACE 186,
  antiviral 160, ESOL 183). Test sets and `test_ids` are unchanged and asserted byte-identical
  by the script; pre-dedup splits kept as `data/splits/<ds>_v2_pre_dedup_2026-10-08.json`.
  **This applies to the LLM runs as well** (`Run_ICL.jl` reads the same JSONs; dry run and
  `check_prompt_parity.py` pass with the new splits).
- **No cross-validation.** Test and training molecules are disjoint; the order of the training
  pool is a seeded permutation (seed 20260907) and the in-context set of size N is its first N
  molecules, so the sets are nested and the training block of a (dataset, size) cell is
  identical for all 150 test molecules and all repeats (prompt caching).
- **Context draws (2026-10-13, section 15).** Besides this main split (draw 1) there are two
  independent nested context draws `data/splits/<ds>_v2_draw2.json` / `_draw3.json`: same test set,
  sizes and deduplicated pool, independently permuted pool order. Used at N = 30 / 60 / 120 only.
- **Test-set rule:** random (same permutation), 150 molecules. For Lipophilicity, BACE and
  antiviral only molecules ABSENT from every searched pre-training corpus are eligible
  in every form (verbatim / canonical SMILES, InChIKey) and every one of the 7 indexes.
  **Final since 2026-09-09** (`make_splits.py --require all`): Lipophilicity 20 of 170
  candidates excluded (16 by InChIKey only), BACE and antiviral none. ESOL: first 150 of the
  permutation, no corpus filter. See CONTAMINATION_CHECK_PLAN section 7.

## 3. Contamination checks (docs/CONTAMINATION_CHECK_PLAN.md)

1. Corpus search over 7 open indexes for all molecules of the four datasets (running).
2. Zero-shot recall test (`src/julia/Run_ZeroShot_TestSet.jl`) on the 150 test molecules per dataset
   for the six panel models (panel of 2026-10-09), prompt and scoring of the memorization paper, reasoning effort
   `low` (the main experiment's setting), max_tokens 32000, emitted tokens recorded.
   3600 calls (seven-model panel: 4200; old five-model panel: 3000 calls, roughly $40-80 at low effort).
   None of the six panel models was in the 45 audit, so this is their first contamination
   measurement. Started 2026-10-09 (running). The zero-shot prompt and its dataset words
   (`src/julia/Prompts_ZS.jl`, `src/memo45/datasets_45.json`) are deliberately left in the 45
   wording, so the verdicts stay comparable with the 45 paper; the ICL prompt revision of
   2026-10-09 (section 4) does not apply to it.

**Result (seven-model panel, `results/contamination/zeroshot/cells.csv`: 4,200 calls, $6.65, parse 100 %):** Lipophilicity, BACE and antiviral clean for all seven panel models (21/21 cells); ESOL contaminated for six of seven (Sonnet 5.5 / GPT-6.1 Sol / GPT-6 Astra / Gemini 3.8 Flash heavy, Haiku 5.5 / GPT-6 Luna partial) and clean for GLM 5.3 Flash. *Correction 2026-10-14: this paragraph read "$3.37, 3,600 calls ... five of six" until 2026-10-14; those were the six-model values of the run of 2026-10-09, before GPT-6 Astra (600 calls, $3.28, 2026-10-13) was added. The file values above are the ones to quote.* **Addendum 2026-10-13:** GPT-6 Astra (seventh panel model) clean on the three clean sets, ESOL heavy (hit3 18.6 %); screen totals now 4,200 calls / $6.65 (CONTAMINATION_CHECK_PLAN.md section 9). Full table: docs/CONTAMINATION_CHECK_PLAN.md section 9; per-molecule hit flags in `results/contamination/zeroshot/<ds>_v2test150_permolecule.csv`.

## 4. Main experiment (`src/julia/Run_ICL.jl`, `src/julia/Prompts_ICL.jl`, `src/julia/Common.jl`)

- **Implementation (decision 2026-09-08):** Julia, on the in-house LLMUtils.jl library via
  `LLMs.jl` (in this repository the library is shipped in `src/julia/lib/`, with a reduced `LLMs.jl` that only loads it and reads the key from `OPENROUTER_API_KEY`; the in-house version also held the model handles) as in the memorization paper's
  `Run_ZeroShot.jl` and the transfer-learning paper's `Run_*_Experiments.jl`; prompts as a
  placeholder dictionary filled by `fill_prompt` (`Prompts_ICL.jl`). `Common.jl::ask_gpt_full`
  reuses the library's request construction, cost limit and usage accounting but returns the
  full response so that prompt / cached / completion / reasoning tokens, cost, finish_reason
  and provider are stored per call. The Python module `src/icl_common.py` is the prompt
  reference; `src/check_prompt_parity.py` verifies both produce identical messages (passes for
  all four datasets). Analysis stays in Python (`src/analyze_icl.py`).
- **Protocol:** io, blinding level 1 (real property name and units, real values, real SMILES).
  Three messages: system role (v1 wording, revised 2026-10-09, see below) -> training block as its own user message, one
  record per molecule (`SMILES: ...` / `<label>: ...`, values rounded to 10 decimals to remove
  float artefacts of the source CSVs), tagged `cache_control: ephemeral` -> task message with
  the test SMILES and the v1 JSON contract `{"prediction", "uncertainty"}` (explicit 1-sigma in
  target units). ESOL label: "Measured log solubility (log(mol/L))" -- the file also carries
  the ESOL-predicted value, which is not used.
- **Prompt revision 2026-10-09 (project lead).** v2 is a new study, so the v1 wording is not
  binding; the revision makes the task statement chemically complete and the uncertainty
  well defined. Changes (identical in `src/julia/Prompts_ICL.jl` / `Common.jl::DATASETS` and the
  Python reference `src/icl_common.py` / `src/common.py::DATASETS`):
  1. System prompt: "You have to use your knowledge and abilities to analyze the training datas
     molecules and the patterns and relationships between molecular properties and the
     <label>" -> "Use your chemical knowledge and the relationships between molecular structure
     and the <label> in the training molecules to make an accurate prediction." The `<medium>`
     placeholder is gone ("of organic molecules based on their SMILES strings").
  2. Uncertainty sentence defines 1 sigma: "Also report your 1-sigma uncertainty for this
     prediction, in the same units as the <label>: the standard deviation of your predictive
     distribution, so that the measured value should fall within prediction +/- uncertainty
     about 68 % of the time." ("measured value", not "true value": experimental noise belongs
     inside sigma, which is what calibration is scored against.)
  3. JSON instructions shortened (single fenced block with EXACTLY `{"prediction", "uncertainty"}`;
     hard requirements: prediction = the label, "`uncertainty` is that 1-sigma value (a
     non-negative number)", the block is the last thing in the response).
  4. Training header: "<n_train> molecules from the same source as the test molecule, with their
     SMILES and known <label> values:" (no third "dataset").
  5. Dataset words: lipophilicity task "lipophilicity (octanol/water distribution coefficient
     logD at pH 7.4)" (logD is a distribution, not a partition coefficient), label "logD (pH 7.4)".
     Antiviral: the ASAP/Polaris 2025 value is the biochemical SARS-CoV-2 Mpro pIC50 column
     (`../37_TransferLearning_Paper/RepoICML/src/prepare_potency_dataset.py`), so task / predict
     "SARS-CoV-2 main protease (Mpro) inhibitory potency (pIC50)" (parallel to BACE; "antiviral
     potency" suggested a cell-based EC50 and echoed the challenge name), label "SARS-CoV-2 Mpro
     pIC50", expertise "... SARS-CoV-2 main protease (Mpro) inhibitor potency prediction".
     ESOL: task / predict "measured aqueous solubility logS (log10 of the solubility in mol/L)",
     label "measured logS (log10 mol/L)" -- keeps "measured" (the file also carries an
     ESOL-predicted column) but no longer repeats the Delaney column header phrase "measured log
     solubility"; log base stated. BACE unchanged.
  6. Training values are printed with a fixed 2 decimals (`fmt_value`, Julia `@sprintf("%.2f")`,
     Python `f"{v:.2f}"`; checked identical on all 7,847 values, no "-0.00"). Before, the digit
     count was a source signature: MoleculeNet bace.csv has 7-decimal pIC50 values (1153 of
     1355 pool values), the antiviral file 10-decimal stereo-averaging artefacts (14 values),
     Delaney float artefacts. Rounding error <= 0.005 against RMSEs of 0.5-1.3. Truth values for
     scoring stay unrounded. Not changed: SMILES are verbatim from the source files (e.g. BACE
     `[NH2+]`), a blinding-level question, not a prompt one.
  Review (Opus 5.5, 2026-10-09): no dataset or benchmark name in the ICL prompt, no test labels,
  test/training pools disjoint by id, canonical isomeric and non-isomeric SMILES.
  
  Parity check (`python src/check_prompt_parity.py`): IDENTICAL for all four datasets after the
  revision. The zero-shot contamination prompt is unchanged (section 3). The 2026-10-08 probe
  measurements (section 9) were taken with the pre-revision wording; the revision changes the
  prompt by a few dozen tokens of a ~50-90k-token prompt, so the measured token / cost figures
  still apply.
- **Panel (since 2026-10-09):** six models, registry `src/registry/models_v2.json`
  (`PANEL` in Common.jl / common.py = its `in_panel` entries in this order). Prices USD per
  Mtok from the OpenRouter `/endpoints` of the endpoint actually reached (flex where it
  exists), fetched 2026-10-08. Pinned providers are sent with `provider.allow_fallbacks =
  false`, so a call fails rather than landing on another host, quantisation or price. The
  Anthropic models are pinned to Anthropic's own endpoint (regional Vertex / Azure / Bedrock
  hosts bill 1.1x and would split the prompt cache). For flex, the order must not name the
  standard endpoint: `google-vertex/global` IS the standard Vertex endpoint (flex:
  `google-vertex/global/flex`). The effort probe compares billed with list cost per call
  (`cost_ratio`) to show whether flex / the pinned price was actually applied.

  Panel of 2026-10-09 (registry order). "out tok @low" = measured completion tokens per call
  (incl. reasoning) at effort `low`: ICL = mean of the two ICL-probe calls (lipophilicity,
  N = 1000, 2026-10-08), ZS = median of the effort-probe calls on the zero-shot prompt.
  Projection = main run (4 datasets x 6 sizes x 150 molecules, `low`) from
  `results/meta/probe_summary.csv` (`proj_main_R1` / `proj_main_R3`; measured output tokens,
  call-2 cache share and tokenizer factor, cache-write premiums not modelled).

  | tag | model | OpenRouter id | routing | tier | in / cache read / out (USD/Mtok) | out tok @low ICL (rsn) / ZS | proj. main R = 1 / R = 3 | open weights |
  |---|---|---|---|---|---|---|---|---|
  | haiku55 | Claude Haiku 5.5 | anthropic/claude-haiku-5.5 | anthropic, no fallback | std | 0.10 / 0.01 / 0.50 | 252 (102) / 5 | $1.8 / $5.5 | -- |
  | sonnet55 | Claude Sonnet 5.5 | anthropic/claude-sonnet-5.5 | anthropic, no fallback | std | 2 / 0.10 / 10 | 313 (0) / 231 | $30.1 / $90.3 | -- |
  | gpt61sol | GPT-6.1 sol | openai/gpt-6.1-sol | openai (-> openai/flex), no fallback | flex | 1 / 0.05 / 5 | 324 (292) / -- (flex 502 at `low`) | $12.4 / $37.2 | -- |
  | gem38flash | Gemini 3.8 Flash | google/gemini-3.8-flash | google-ai-studio (-> google-ai-studio/flex), no fallback | flex | 0.375 / 0.0375 / 1.875 | 763 (733) / 318 | $8.9 / $26.8 | -- |
  | gpt6luna | GPT-6 luna | openai/gpt-6-luna | openai (-> openai/flex), no fallback | flex | 0.05 / 0.005 / 0.25 | 500 (468) / 860 | $0.9 / $2.8 | -- |
  | glm53flash | GLM 5.3 Flash | z-ai/glm-5.3-flash | z-ai (z-ai/fp8), no fallback | std | 0.15 / 0.03 / 0.50 | 132 (104) / 38 | $2.6 / $7.7 | zai-org/GLM-5.3-Flash |
  | **total** | | | | | | | **$56.8 / $170.3** | |

  Dropped 2026-10-09 (kept in the registry, `in_panel: false`):

  | tag | model | OpenRouter id | routing | tier | in / cache read / out | reason |
  |---|---|---|---|---|---|---|
  | opus55 | Claude Opus 5.5 | anthropic/claude-opus-5.5 | anthropic, no fallback | std | 4 / 0.20 / 20 | Anthropic's safety classifier refuses the ICL prompt from N >= 250 (`content_filter`; answers at N = 30) |
  | dsv41flash | DeepSeek V4.1 Flash | deepseek/deepseek-v4.1-flash | deepseek, no fallback | std | 0.15 / 0.003 / 0.60 (off-peak) | HTTP 404: blocked by the OpenRouter account's privacy setting (providers that train on paid prompts) |
  | kimik3 | Kimi K3 | moonshotai/kimi-k3 | moonshotai (mxfp4), no fallback | std | 3 / 0.30 / 15 | too expensive ($69 / $206 main run at R = 1 / 3, measured) |
  | mimo26pro / mimo26flash | MiMo V2.6 Pro / Flash | xiaomi/mimo-v2.6-pro / -flash | xiaomi, no fallback | std | 0.43 / 0.0036 / 0.87 and 0.14 / 0.0028 / 0.28 | reasoning none-or-unbounded: 5 tokens at `none`; Flash 32k (cap) at `minimal` and `low`, Pro no answer within 900 s at `minimal` (section 10) |

  Further reserve (`in_panel: false`, probed 2026-10-08): GPT-5.6 terra (flex; panel until 2026-10-08 evening),
  Opus 5, GPT-5.6 sol (flex), Gemini 3.6 Flash (flex), Grok 4.5, Grok 4.7, GLM 5.3 (z-ai),
  Qwen3.8 2.4T-A95B (alibaba), and the additions of 2026-10-08 (prices from the OpenRouter
  `/endpoints`, snapshot `results/meta/openrouter_endpoints_2026-10-08_additions.json`; every one
  pinned to its vendor's own endpoint, no fallback; gpt6luna and glm53flash, added the same day,
  are panel members since 2026-10-09 and listed above):

  | tag | model | OpenRouter id | routing | tier | in / cache read / out | open weights |
  |---|---|---|---|---|---|---|
  | gpt6astra | GPT-6 astra (probe only, too expensive for the main run) | openai/gpt-6-astra | openai (-> openai/flex) | flex | 5 / 0.5 / 25 | -- |
  | mistrall4 | Mistral Large 4 | mistralai/mistral-large-4-0 | mistral | std | 0.68 / 0.07 / 2.09 (listed with discount 0.5) | -- |
  | hy4preview | Hy4 preview | tencent/hy4-preview | tencent (tencent/fp8) | std | 0.75 / 0.0378 / 2.25 (16-24 UTC; 00-16 UTC 1.11x) | tencent/Hy4-preview |
  | step5preview | Step 5 Preview | stepfun/step-5-preview | stepfun (stepfun/fp8) | std | 1.00 / 0.05 / 2.70 | -- |
  | dsv4pro | DeepSeek V4 Pro 0813 | deepseek/deepseek-v4-pro-0813 | deepseek | std | 0.66 / 0.022 / 1.98 (off-peak) | deepseek-ai/DeepSeek-V4-Pro-0813 |
  | qwen3827b | Qwen3.8 27B | qwen/qwen3.8-27b | alibaba | std | 0.425 / 0.085 / 2.55 | Qwen/Qwen3.8-27B |

  ESOL contamination is known from 45 for none of the six panel models (45 measured terra and
  Kimi K3, both now out of the panel); the zero-shot check (section 3) decides. At fetch time (2026-10-08, ~19 UTC) the
  `openai/flex` endpoint of GPT-6.1 sol reported status -5 and 0 % uptime over the last 30 min
  (62 % over one day); see the probe results (section 9) for what that meant in practice.
- **Reasoning:** effort `low` for every model (decision 2026-09-08; not a token budget:
  endpoints map budgets to levels and interpret them differently). Emitted reasoning tokens, completion tokens, prompt
  tokens and cached prompt tokens are stored per call. Completion cap 32000. Whether the
  requested effort changes what a model emits is MEASURED, not assumed, by the
  effort-dosability probe `src/julia/Probe_Models.jl` (one zero-shot call per model x level;
  levels supported, HTTP 400 rejections, monotonicity none<low<high of the emitted reasoning
  tokens, suggested `reasoning_levels` for the registry, entered by hand).
- **Sampling:** provider defaults (temperature / top_p omitted, i.e. temperature 1.0 for every
  model; decided 2026-10-08, confirmed for the six-model panel 2026-10-09); the OpenAI reasoning
  family rejects other values anyway.
- **Repeats (decided 2026-10-08, confirmed 2026-10-09):** R = 1 first, extended to R = 3 later -> UC1 (mean
  self-reported sigma) from the start, UC3 (variance over repeats) once R = 3 exists.
  `ICL_REPEATS` defaults to 1. The runner indexes repeats 0..R-1 and on restart skips every
  (mol_id, repeat) pair already stored in the cell file (`Common.jl::existing_pairs`,
  `Run_ICL.jl::run_cell`), so a later run with `ICL_REPEATS=3` on the same cells only adds
  repeats 1 and 2 (verified in the code 2026-10-08).
- **Cells:** 4 datasets x 6 models x 6 sizes = 144 cells, 21,600 calls at R = 1 (64,800 at R = 3)
  (seven-model panel: 168 cells, 25,200 / 75,600 calls).
- **Output:** `results/icl/<ds>__<tag>__N<size>__low.jsonl`, one record per call with the
  full response text; resumable (restart skips stored (molecule, repeat) pairs).

## 5. Sub-experiment: reasoning ladder

Same runner: `ICL_REASONING=none,minimal,low,medium,high ICL_SIZES=min,max`, i.e. the smallest
(30) and largest (1000 / 844 / 976) in-context size only. Levels an endpoint rejects (in 45:
Grok 4.5 and Gemini 3.5 Flash rejected `none`) are recorded as unsupported; per model only the
levels the effort probe (`Probe_Models.jl`) finds supported AND dosable (`reasoning_levels` in
the registry) are worth running -- 45 showed several Chinese endpoints (Kimi K2.6, GLM 5 / 5.2,
Qwen3.5, DeepSeek V4 Pro) ignore the effort entirely. The `low` cells are shared with the main
experiment. Ladder composition still under consideration. Upper bound: 4 datasets x 6 models x
2 sizes x 4 extra levels = 192 cells, 28,800 calls at R = 1 / 86,400 at R = 3 (seven-model
panel: 224 cells, 100,800 calls at R = 3). `none` is rejected by gpt61sol, gem38flash, sonnet55
and glm53flash (section 9), so the real ladder is smaller.

## 6. Cost (src/estimate_icl_cost.py, results/icl_cost_estimate.csv)

Prices of the v2 registry (fetched 2026-10-08, table in section 4); assumptions: 55 tokens per
training record for every dataset (measured with tiktoken: lipophilicity 47, BACE 64, antiviral
52, ESOL 35 tokens per record; Claude tokenizers count more -- BACE at N = 1000 is about 80-87k
prompt tokens there, still below Haiku 5.5's 100k long-context surcharge; GPT-5.6 surcharges
above 272k), 450 fixed prompt tokens, 80 % of prompt tokens billed at the endpoint's
cache-read price at N >= 120 (`price_cache_read` of the registry; cache-write premiums not
modelled), **1200 output tokens per call at `low`** (v1 measured 685-5356 across providers for
"medium"; this single number dominates the total), ladder levels none 150 / minimal 400 /
medium 2500 / high 6000 output tokens.

`python src/estimate_icl_cost.py --out-main 1200 --main-level low --repeats 3 --cache-frac 0.8`
(R = 2: `--repeats 2 --csv results/icl_cost_estimate_R2.csv`), USD (panel with terra, before the
swap; the gpt61sol panel follows below the table):

| | Haiku 5.5 | DeepSeek V4.1 Flash | Gemini 3.8 Flash | GPT-5.6 terra | Sonnet 5.5 | Kimi K3 | Opus 5.5 | total |
|---|---|---|---|---|---|---|---|---|
| main experiment, `low`, R = 3 | $13 | $15 | $48 | $140 | $239 | $381 | $478 | **$1,314** |
| main experiment, `low`, R = 2 | $8 | $10 | $32 | $93 | $159 | $254 | $319 | **$876** |
| reasoning ladder (none / minimal / medium / high at min & max size), R = 3 | $29 | $34 | $107 | $318 | $540 | $856 | $1,080 | **$2,962** |
| reasoning ladder, R = 2 | $19 | $23 | $71 | $212 | $360 | $570 | $720 | **$1,975** |
| of which `high` alone, R = 3 | $14 | $17 | $52 | $160 | $269 | $416 | $539 | $1,467 |

**Panel with gpt61sol (2026-10-08 evening), same assumptions**
(`--models haiku55,sonnet55,opus55,gpt61sol,gem38flash,dsv41flash,kimik3,terra`, CSVs
`results/icl_cost_estimate_2026-10-08_gpt61sol_R3.csv` / `_R1.csv`): gpt61sol main run $119 at
R = 3 / $40 at R = 1 vs terra $140 / $47 (cheaper output 5 vs 6 and cache read 0.05 vs 0.10);
panel main total **$1,292 at R = 3** (was $1,314) and **$430 at R = 1**; ladder at R = 3 $2,911.
The measured replacement of these assumptions is in section 9; the measured estimate for the
six-model panel of 2026-10-09 is **$57 main run at R = 1 / $170 at R = 3** (ladder $256 / $768),
section 10.

**Main uncertainties of this estimate (review 2026-10-08):**

1. **Output tokens per call.** 1200 at `low` was the dominant unmeasured assumption (measured 2026-10-08: 250-760 for the working panel models, section 9); output
   tokens dominate for the expensive models, so the estimate scales nearly linearly with the
   true `low` trace length.
2. **Kimi K3 cache share is unmeasured** (resolved 2026-10-08: 99.8 % cached on the second call, section 9). Moonshot's endpoint lists no implicit caching. Without
   cache hits Kimi K3's main run costs $779 instead of $381 and the R = 3 main total is about
   **$1,712** instead of $1,314. (With no cache hits for any model, `--cache-frac 0`: $2,769.)
3. Tokens per record differ by dataset (35-64 with tiktoken, more on Claude tokenizers); the
   uniform 55 is a middle value, not an upper bound.

**Mandatory measurement before any commitment:** `ICL_PROBE=1 ICL_SIZES=max julia src/julia/Run_ICL.jl`
(two identical calls per model at the largest size of the first dataset -- without `ICL_SIZES=max` the
probe uses N = 30, where nothing is cached -- the second one to see cache hits) to
measure trace length, prompt and cached tokens and the billed cost per call; then re-run the
estimate with `--out-main <measured>` (and `--cache-frac` / `--tok-per-record` from the
measurement). No paid run of the main experiment or the ladder before that. Levers for the ladder: drop `high` (saves ~50 %),
R = 2, restrict it to the models the effort probe finds dosable, or run it on two datasets only.
Ladder composition still open.

## 7. Analysis (`src/analyze_icl.py`, then `src/v1reuse/uq_methods_eval.py`)

Per cell: RMSE / MAE / Spearman(truth, prediction); rho(UC1, |err|), rho(UC3, |err|);
Gaussian PICP / MPIW at 68 and 90 %, NLL, CRPS; parse and truncation rate; median reasoning
tokens; cache share; cost. Then the v1 calibration suite (PIT / ECE, ENCE, variance-scaling,
isotonic and conformal recalibration) and paired bootstrap CIs. Contamination layer: every
test molecule carries corpus-presence flags and the zero-shot hit flags of each panel model;
metrics are reported on all molecules and on the never-recalled subset, and on ESOL the
recalled vs non-recalled molecules are compared per model.

### Classical baselines (`src/baselines/`, 2026-10-08, rerun on the deduplicated splits, no API cost)

Run on exactly the LLM cells of the **deduplicated** splits (section 2): per dataset and ladder
size N, fit on the first N molecules of `train_order_ids`, predict the 150 `test_ids`
(4 datasets x 6 sizes; seeds 0, 1, 2 for tabpfn / rf / rf_desc, seed 0 reported; 240 cell files).
The pre-dedup run is kept in `results/baselines/pre_dedup_2026-10-08/` (superseded).

- **tabpfn** -- TabPFNRegressor **v3.5** (`create_default_for_version(ModelVersion.V3_5)`,
  checkpoint `tabpfn-v3.5-20260909`, tabpfn 9.1.0, n_estimators "auto", CPU) on RDKit 2D
  descriptors (all 217 of `descList` in rdkit 2026.03; NaN/inf -> NaN, clip +-1e8; per cell:
  columns constant or > 50 % NaN on the N training rows dropped, NaN imputed with the
  training-row median -> 159-201 features (`results/baselines/tabpfn__*.meta.json` n_features; *corrected
  2026-10-14 from "160-201"*); no normalisation, TabPFN preprocesses itself).
  Prediction = mean of the predictive distribution; **sigma = (q84 - q16) / 2**; the quantiles
  q05/q16/q25/q50/q75/q84/q95 are stored, so intervals can also be read off directly
  (`PICP68_q` = [q16, q84], `PICP90_q` = [q05, q95]). Diagnostic `sd_full` (SD of the full bar
  distribution) is 3-17 % larger than sigma (heavier than Gaussian tails). Vendor statement
  (model card Prior-Labs/tabpfn_3_5, read 2026-10-08): TabPFN-3.5 "is trained purely on synthetic
  tabular tasks", i.e. no pretraining exposure to these (or any real) datasets -- this refers
  to v3.5 specifically, not to earlier TabPFN versions.
- **rf** -- RandomForestRegressor, 200 trees, Morgan r=2 / 2048 bits (as v1); sigma = std over
  the tree predictions (ddof = 1).
- **rf_desc** -- the same RF on the same per-cell descriptor matrix as tabpfn, so that the
  learner effect (tabpfn vs rf_desc) is separated from the feature effect (rf_desc vs rf).
- **trivial** -- training mean, sigma = training std (sanity floor).
- **Leakage guard:** the label vector given to every fit has the test labels set to NaN; the
  selfcheck replaces every label outside the cell's training rows (test and unused pool) by junk
  and verifies identical predictions for all four methods, and that every cell's training ids are
  `train_order_ids[:N]` and its test ids `test_ids`.
- **Env:** conda env `uc44` (Python 3.11, torch 2.14.1+cpu, tabpfn 9.1.0, rdkit 2026.03.6,
  scikit-learn 1.9.1; `env/uc44.yml`, exact pins `env/requirements-uc44.lock.txt`); all cells
  were produced there. Weights (`tabpfn-v3.5-20260909.safetensors`, 876 MB) from Hugging Face
  `Prior-Labs/tabpfn_3_5` into `%APPDATA%/tabpfn` after TabPFN's license check, which reads
  `TABPFN_TOKEN` (from the env or the git-ignored `.env` in the repository root, at runtime only). **Side effect:** the
  tabpfn library itself caches the verified token in `~/.cache/tabpfn/auth_token` (outside this
  repo; delete it to remove the local copy). The GPU is not used (CPU torch wheel).
- **Wall clock (CPU, seed 0, 24 cells each):** tabpfn 691 s (9-11 s at N = 30, 62-98 s at the
  top rung); rf 31 s; rf_desc 13 s; trivial 0 s. Seeds 1 + 2: tabpfn 1457 s, rf 60 s,
  rf_desc 30 s.
- **Commands:** `conda activate uc44`;
  `python src/baselines/run_baselines.py --methods tabpfn,rf,rf_desc,trivial --datasets all --sizes all --seeds 0`
  (skips existing cells unless `--force`); `python src/baselines/selfcheck.py --tabpfn`;
  `python src/baselines/analyze_baselines.py` -> `results/baselines/summary.csv`.
- **Metrics / columns:** names as in `analyze_icl.py` (RMSE, MAE, spearman_pred, rho_UC1_err
  with UC1 := the method's sigma, mean_UC1, PICP/MPIW 68/90, NLL, CRPS), plus ECE, Winkler_90 and
  the recalibration columns of `v1reuse/uq_methods_eval.py`. Key columns: `method` corresponds
  to `model` and `seed` to `reasoning` of `results/icl/cells.csv` (`dataset`, `size` shared).
  ENCE / sigma_CV are computed by a local copy of v1's `ence` that sorts sigma with
  `kind="stable"` (v1's default quicksort orders tied sigmas differently across numpy builds,
  which made ENCE environment-dependent). `summary.csv` was produced in the **base env**
  (python 3.11.7, numpy 1.26.4, scipy 1.11.4, sklearn 1.5.2; recorded in its `env` column);
  rerunning the analysis in `uc44` gives identical values for every metric.

Seed 0, RMSE / Spearman(truth, prediction) / rho(sigma, |err|) per size (top rung = 1000 /
1000 / 844 / 976):

| dataset | method | 30 | 60 | 120 | 250 | 500 | top |
|---|---|---|---|---|---|---|---|
| lipophilicity | tabpfn | 1.17 / 0.39 / 0.13 | 1.01 / 0.47 / 0.12 | 0.94 / 0.54 / 0.24 | 0.83 / 0.62 / 0.38 | 0.72 / 0.70 / 0.39 | 0.62 / 0.79 / 0.36 |
| lipophilicity | rf_desc | 1.22 / 0.39 / 0.07 | 1.08 / 0.38 / 0.16 | 0.94 / 0.55 / 0.25 | 0.88 / 0.60 / 0.37 | 0.81 / 0.66 / 0.33 | 0.77 / 0.68 / 0.35 |
| lipophilicity | rf | 1.33 / 0.11 / 0.08 | 1.17 / 0.19 / 0.06 | 1.05 / 0.34 / 0.13 | 0.98 / 0.38 / 0.27 | 0.95 / 0.44 / 0.19 | 0.90 / 0.50 / 0.28 |
| bace | tabpfn | 1.15 / 0.48 / 0.44 | 1.01 / 0.61 / 0.23 | 1.02 / 0.61 / 0.36 | 0.91 / 0.70 / 0.25 | 0.83 / 0.74 / 0.26 | 0.67 / 0.82 / 0.26 |
| bace | rf_desc | 1.16 / 0.48 / 0.36 | 1.13 / 0.53 / 0.23 | 1.12 / 0.57 / 0.32 | 0.94 / 0.69 / 0.32 | 0.85 / 0.74 / 0.38 | 0.75 / 0.81 / 0.28 |
| bace | rf | 1.21 / 0.35 / 0.37 | 1.13 / 0.53 / 0.34 | 1.07 / 0.60 / 0.34 | 1.02 / 0.65 / 0.34 | 0.89 / 0.73 / 0.35 | 0.74 / 0.81 / 0.38 |
| antiviral | tabpfn | 0.85 / 0.62 / 0.35 | 0.85 / 0.62 / 0.33 | 0.79 / 0.67 / 0.36 | 0.72 / 0.74 / 0.43 | 0.69 / 0.76 / 0.44 | 0.61 / 0.80 / 0.42 |
| antiviral | rf_desc | 0.84 / 0.60 / 0.31 | 0.86 / 0.60 / 0.33 | 0.79 / 0.68 / 0.37 | 0.71 / 0.74 / 0.44 | 0.69 / 0.75 / 0.42 | 0.64 / 0.78 / 0.42 |
| antiviral | rf | 0.97 / 0.59 / 0.36 | 0.93 / 0.59 / 0.43 | 0.82 / 0.63 / 0.39 | 0.79 / 0.69 / 0.38 | 0.75 / 0.72 / 0.42 | 0.66 / 0.73 / 0.51 |
| esol | tabpfn | 0.94 / 0.87 / 0.26 | 0.90 / 0.90 / 0.33 | 0.78 / 0.91 / 0.47 | 0.62 / 0.95 / 0.42 | 0.56 / 0.96 / 0.53 | 0.51 / 0.96 / 0.58 |
| esol | rf_desc | 0.91 / 0.85 / 0.09 | 0.94 / 0.87 / 0.12 | 0.88 / 0.87 / 0.10 | 0.78 / 0.90 / 0.22 | 0.71 / 0.92 / 0.23 | 0.68 / 0.93 / 0.40 |
| esol | rf | 1.69 / 0.53 / 0.08 | 1.85 / 0.48 / -0.31 | 1.69 / 0.58 / 0.02 | 1.47 / 0.68 / 0.02 | 1.21 / 0.79 / 0.22 | 1.13 / 0.82 / 0.35 |

Trivial-predictor RMSE (floor): lipophilicity 1.10-1.33, BACE 1.34-1.35, antiviral 1.18-1.19,
ESOL 1.96-2.04. Seed SD of RMSE <= 0.025 for every method. Coverage: tabpfn Gaussian PICP68
0.60-0.83, PICP90 0.83-0.93 (over-confident at small N on lipophilicity; ECE <= 0.05 in 19 of 24
cells, 0.14 at lipophilicity N = 30); rf_desc PICP68 0.53-0.79 / PICP90 0.85-0.95; rf PICP68
0.51-0.75 / PICP90 0.68-0.94.

Reading: (i) features dominate on ESOL (rf -> rf_desc: top-rung RMSE 1.13 -> 0.68); the learner
adds the rest (rf_desc -> tabpfn: 0.68 -> 0.51) and makes sigma more informative (rho 0.40 ->
0.58); (ii) on antiviral and BACE rf_desc is about equal to tabpfn up to N = 500 (BACE N = 60-120
excepted), tabpfn ahead only at the top rung; (iii) on lipophilicity tabpfn is clearly ahead from
N = 250 on (0.62 vs 0.77 at N = 1000). Effect of the dedup vs the pre-dedup run: N = 30 / 60
cells identical (and N = 120 except lipophilicity); elsewhere |delta RMSE| <= 0.03. The largest
shift is ESOL at the top rung (tabpfn 0.489 -> 0.511, rf 1.095 -> 1.125): the two removed ESOL
duplicates carried the identical label of their test twin, i.e. they had been a small label leak.

MLP ensemble and Chemprop ensemble of v1: run 2026-10-13, together with a Tanimoto GP and kNN, see the next subsection.


(Note 2026-10-12: `results/baselines/summary.csv` was regenerated after `analyze_icl.gaussian_metrics` switched PICP68 to z = 1.0; the PICP68 values quoted in the tables of this section may be 1/150 low in a few cells -- the CSV is authoritative.)

### Additional uncertainty baselines (2026-10-13): GP, deep ensembles, kNN (no API cost)

Why: the paper's question is whether a sigma ranks the errors at small N (active-learning start),
so the classical side needs methods whose sigma is principled, not only TabPFN and the RF tree
spread. Same cells, files, leakage guard and metrics as above (`run_baselines.py --methods
gp_tanimoto,mlp_ens,chemprop_ens,knn_tanimoto`; per-cell CSV + `.meta.json` with all fitted
hyperparameters / epochs / versions). Seed 0 for every cell (draw 1 all sizes, draws 2 and 3 at
N = 30 / 60 / 120); gp_tanimoto and knn_tanimoto also seeds 1 and 2. 384 new cells (gp 144,
knn 144, mlp 48, chemprop 48); `summary.csv` now has 720 rows, the 336 previous rows unchanged in
content and order (new methods are appended to the method order).

- **gp_tanimoto** -- exact GP regression in numpy / scipy on the Morgan r=2 / 2048-bit
  fingerprints (`features.morgan_matrix`, cache of rdkit 2026.03.6), kernel
  k = a * Tanimoto(x, x') + s2 * delta (Tanimoto on bits), constant mean = training mean (not
  fitted). a and s2 maximise the log marginal likelihood: L-BFGS-B on (log a, log s2) with the
  analytic gradient (checked against finite differences, 1e-6), bounds a in [1e-3, 1e2] x var(y),
  s2 in [1e-4, 1e1] x var(y), 14 starts ((var, 0.1 var), a 3 x 3 grid a in {0.01, 0.3, 10} x
  s2 in {0.001, 0.05, 1} x var(y), 4 random starts from `default_rng(seed)`), jitter 1e-8;
  prediction = posterior mean, **sigma = posterior predictive SD including the noise s2**. A first
  run with only 5 starts missed the better optimum for one seed in 2 of 48 cells, so all GP cells
  were rerun with the grid (`--force`); now all three seeds give identical fits in every cell, i.e.
  the seed is irrelevant for the GP. At N = 30 the marginal likelihood is bimodal and nearly flat
  (optima < 0.3 nats apart): an interpolating solution (s2 at its lower bound; 14 of 48 cells
  over all N, mostly N <= 60) and a noise-only solution (a at its lower bound = the trivial
  predictor: training mean, sigma = training SD), which ML-II picks in 4 of the 12 N = 30 cells
  (lipophilicity draws 1 and 2, BACE draw 2, ESOL draw 2). The mean off-diagonal Tanimoto
  similarity of a training set is only 0.08-0.33, so amplitude and noise are hard to separate.
- **mlp_ens** -- deep ensemble of 5 MLPs (torch 2.14 CPU, 4 threads): raw 2048 bits ->
  256 -> 256 -> 1, ReLU, dropout 0.1, MSE on targets standardised with the training mean / SD,
  Adam lr 1e-3, batch min(64, N); stopping rule of v1 (sklearn `MLPRegressor` defaults: no
  validation split, at most 500 epochs, stop after 10 epochs without a training-loss gain > 1e-4;
  27-103 epochs used); member k seeded 100 * seed + k (init, dropout, shuffling); prediction =
  member mean, **sigma = member SD (ddof = 1)**. Deviation from v1
  (`archive/v1_2026-09_contaminated/src/baselines_rf_mlp.py`, which used sklearn MLPRegressor
  with five different architectures (256,128) / (128,128) / (256,256) / (128,64) / (200,100),
  bootstrap bagging, StandardScaler on the bits and unscaled targets): one architecture, no
  bagging (standard deep ensemble), raw bits, standardised targets. A smoke test of the literal
  v1 recipe on lipophilicity N = 30 gave RMSE 1.69 / mean sigma 0.32 (vs 1.33 / 0.10 here).
- **chemprop_ens** -- the v1 Chemprop recipe (`archive/.../baseline_chemprop.py`), chemprop 2.0.0
  defaults: `BondMessagePassing` (d_h 300, depth 3), `MeanAggregation`, `RegressionFFN`
  (1 x 300), batch norm, Noam LR schedule 1e-4 -> 1e-3 -> 1e-4 with 2 warm-up epochs, targets
  standardised on the training rows, batch min(64, N), **50 epochs** at every size (no
  validation split, as v1), 5 members seeded 100 * seed + k, Lightning `deterministic=True`
  (without it two identical runs differ by up to 1e-3 on CPU; with it bit-identical, same speed);
  prediction = member mean, **sigma = member SD (ddof = 1)**. **Env: the base env** (chemprop
  2.0.0, torch 2.2.2 CPU, lightning 2.2.4, rdkit 2023.03); the conda env `chemprop` exists but is
  empty. The slowest top-rung cell (lipophilicity N = 1000) took 25.2 min (6.0 s / epoch with
  up to three chemprop processes in parallel), below the 30-min limit, so the epochs were not
  reduced anywhere.
- **knn_tanimoto** -- the k = 5 most Tanimoto-similar training molecules (same fingerprints;
  ties at the 5th place broken by training order); prediction = similarity-weighted mean of their
  labels, **sigma = max(SD of the 5 labels (ddof = 1), 0.01)** (the floor was never hit).
  Deterministic: seeds 1 / 2 are identical to seed 0. Median top-1 similarity of a test molecule
  0.36 at N = 30, 0.70 at the top rung.
- **Leakage / integrity:** `selfcheck.py` (uc44: trivial, rf, rf_desc, gp_tanimoto, knn_tanimoto,
  mlp_ens; base env: `selfcheck.py --chemprop --only`) poisons every label outside the
  lipophilicity N = 30 training rows: predictions and sigmas are bit-identical for all methods,
  and the clean refit reproduces the stored cell (<= 4.4e-16, CSV round trip; the comparison is
  gated on the library versions recorded in the meta). All 720 stored cells: 150 finite
  predictions on `test_ids`, training ids = `train_order_ids[:N]` of their draw, sigma > 0, no
  missing cell in the draw-1 seed-0 grid.
- **Wall clock** (CPU, seed 0, draw 1, 24 cells each; Julia ICL runs and up to three of these
  processes in parallel, so the chemprop times are upper bounds): gp_tanimoto 125 s (2 s at
  N = 30, 14-29 s at the top rung; all 144 cells 426 s); knn_tanimoto 2 s; mlp_ens 353 s (3-8 s at
  N = 30, 29-56 s at the top rung; all 48 cells 519 s); chemprop_ens 8,448 s = 2.3 h (40-71 s at
  N = 30, 580-1,511 s at the top rung; all 48 cells 3.1 h). Logs:
  `results/baselines/run_{gp_knn_mlp,gp_rerun,chemprop_ens}_2026-10-13*.log`.

RMSE / rho(sigma, |err|), seed 0, draw 1 (top = 1000 / 1000 / 844 / 976; all methods incl. the
four earlier ones for comparison):

| dataset | method | N = 30 | 60 | 120 | top |
|---|---|---|---|---|---|
| lipophilicity | tabpfn | 1.17 / 0.13 | 1.01 / 0.12 | 0.94 / 0.24 | 0.62 / 0.36 |
| lipophilicity | gp_tanimoto | 1.33 / 0.04 | 1.12 / 0.17 | 1.03 / 0.23 | 0.83 / 0.39 |
| lipophilicity | chemprop_ens | 1.19 / -0.09 | 1.11 / -0.05 | 0.99 / 0.16 | 0.65 / 0.19 |
| lipophilicity | mlp_ens | 1.33 / 0.13 | 1.15 / 0.08 | 1.06 / 0.16 | 0.87 / 0.05 |
| lipophilicity | rf_desc | 1.22 / 0.07 | 1.08 / 0.16 | 0.94 / 0.25 | 0.77 / 0.35 |
| lipophilicity | rf | 1.33 / 0.08 | 1.17 / 0.06 | 1.05 / 0.13 | 0.90 / 0.28 |
| lipophilicity | knn_tanimoto | 1.21 / 0.09 | 1.16 / 0.06 | 1.07 / 0.23 | 0.92 / 0.19 |
| lipophilicity | trivial | 1.33 / - | 1.22 / - | 1.13 / - | 1.10 / - |
| BACE | tabpfn | 1.15 / 0.44 | 1.01 / 0.23 | 1.02 / 0.36 | 0.67 / 0.26 |
| BACE | gp_tanimoto | 1.18 / 0.36 | 1.07 / 0.29 | 1.05 / 0.34 | 0.70 / 0.18 |
| BACE | chemprop_ens | 1.26 / 0.18 | 1.27 / 0.16 | 1.17 / 0.24 | 0.79 / 0.08 |
| BACE | mlp_ens | 1.22 / -0.14 | 1.10 / 0.07 | 1.08 / -0.08 | 0.75 / 0.07 |
| BACE | rf_desc | 1.16 / 0.36 | 1.13 / 0.23 | 1.12 / 0.32 | 0.75 / 0.28 |
| BACE | rf | 1.21 / 0.37 | 1.13 / 0.34 | 1.07 / 0.34 | 0.74 / 0.38 |
| BACE | knn_tanimoto | 1.28 / 0.36 | 1.09 / 0.26 | 1.08 / 0.21 | 0.75 / 0.32 |
| BACE | trivial | 1.34 / - | 1.35 / - | 1.34 / - | 1.34 / - |
| antiviral | tabpfn | 0.85 / 0.35 | 0.85 / 0.33 | 0.79 / 0.36 | 0.61 / 0.42 |
| antiviral | gp_tanimoto | 0.84 / 0.31 | 0.86 / 0.38 | 0.80 / 0.27 | 0.64 / 0.13 |
| antiviral | chemprop_ens | 1.10 / 0.03 | 1.10 / -0.08 | 0.89 / 0.17 | 0.61 / 0.15 |
| antiviral | mlp_ens | 0.88 / 0.02 | 0.90 / 0.26 | 0.86 / 0.11 | 0.68 / 0.12 |
| antiviral | rf_desc | 0.84 / 0.31 | 0.86 / 0.33 | 0.79 / 0.37 | 0.64 / 0.42 |
| antiviral | rf | 0.97 / 0.36 | 0.93 / 0.43 | 0.82 / 0.39 | 0.66 / 0.51 |
| antiviral | knn_tanimoto | 0.92 / 0.16 | 0.95 / 0.09 | 0.86 / 0.20 | 0.72 / 0.37 |
| antiviral | trivial | 1.18 / - | 1.19 / - | 1.18 / - | 1.19 / - |
| ESOL | tabpfn | 0.94 / 0.26 | 0.90 / 0.33 | 0.78 / 0.47 | 0.51 / 0.58 |
| ESOL | gp_tanimoto | 1.78 / -0.01 | 1.71 / 0.00 | 1.59 / 0.07 | 0.96 / 0.09 |
| ESOL | chemprop_ens | 1.48 / -0.14 | 1.47 / 0.06 | 1.01 / 0.18 | 0.53 / 0.21 |
| ESOL | mlp_ens | 1.94 / 0.09 | 1.82 / 0.13 | 1.59 / 0.13 | 1.02 / 0.10 |
| ESOL | rf_desc | 0.91 / 0.09 | 0.94 / 0.12 | 0.88 / 0.10 | 0.68 / 0.40 |
| ESOL | rf | 1.69 / 0.08 | 1.85 / -0.31 | 1.69 / 0.02 | 1.13 / 0.35 |
| ESOL | knn_tanimoto | 1.77 / 0.19 | 1.77 / 0.03 | 1.87 / 0.24 | 1.31 / 0.41 |
| ESOL | trivial | 1.96 / - | 2.04 / - | 2.00 / - | 1.98 / - |

Mean over the three context draws at N = 30 / 60 / 120 (seed 0): RMSE (SD over draws) /
mean rho(sigma, |err|):

| dataset | method | N = 30 | 60 | 120 |
|---|---|---|---|---|
| lipophilicity | tabpfn | 1.10 (0.06) / 0.12 | 1.01 (0.00) / 0.14 | 0.92 (0.04) / 0.26 |
| lipophilicity | gp_tanimoto | 1.17 (0.14) / 0.06 | 1.09 (0.02) / 0.17 | 1.05 (0.02) / 0.17 |
| lipophilicity | chemprop_ens | 1.13 (0.06) / 0.01 | 1.09 (0.02) / 0.01 | 1.02 (0.05) / 0.10 |
| lipophilicity | mlp_ens | 1.18 (0.13) / 0.13 | 1.12 (0.02) / 0.02 | 1.07 (0.02) / 0.08 |
| lipophilicity | rf_desc | 1.18 (0.04) / 0.01 | 1.07 (0.04) / 0.12 | 0.96 (0.02) / 0.15 |
| lipophilicity | rf | 1.18 (0.13) / 0.11 | 1.13 (0.04) / 0.14 | 1.05 (0.02) / 0.17 |
| lipophilicity | knn_tanimoto | 1.15 (0.05) / 0.09 | 1.14 (0.03) / 0.11 | 1.09 (0.04) / 0.15 |
| lipophilicity | trivial | 1.17 (0.14) / - | 1.14 (0.07) / - | 1.11 (0.01) / - |
| BACE | tabpfn | 1.14 (0.03) / 0.28 | 1.08 (0.09) / 0.24 | 0.99 (0.06) / 0.25 |
| BACE | gp_tanimoto | 1.25 (0.11) / 0.36 | 1.17 (0.11) / 0.39 | 1.04 (0.06) / 0.28 |
| BACE | chemprop_ens | 1.27 (0.04) / 0.15 | 1.29 (0.03) / 0.16 | 1.12 (0.09) / 0.17 |
| BACE | mlp_ens | 1.25 (0.06) / -0.18 | 1.17 (0.07) / 0.00 | 1.06 (0.06) / 0.05 |
| BACE | rf_desc | 1.22 (0.07) / 0.16 | 1.15 (0.09) / 0.20 | 1.06 (0.07) / 0.34 |
| BACE | rf | 1.24 (0.07) / 0.23 | 1.21 (0.11) / 0.34 | 1.08 (0.05) / 0.35 |
| BACE | knn_tanimoto | 1.31 (0.06) / 0.22 | 1.20 (0.12) / 0.24 | 1.10 (0.03) / 0.26 |
| BACE | trivial | 1.36 (0.02) / - | 1.36 (0.02) / - | 1.34 (0.00) / - |
| antiviral | tabpfn | 0.83 (0.01) / 0.35 | 0.80 (0.05) / 0.37 | 0.74 (0.04) / 0.39 |
| antiviral | gp_tanimoto | 0.87 (0.03) / 0.24 | 0.84 (0.01) / 0.24 | 0.79 (0.02) / 0.17 |
| antiviral | chemprop_ens | 1.12 (0.08) / 0.04 | 1.07 (0.03) / -0.11 | 0.81 (0.07) / 0.14 |
| antiviral | mlp_ens | 0.89 (0.04) / 0.15 | 0.89 (0.01) / 0.26 | 0.83 (0.03) / 0.20 |
| antiviral | rf_desc | 0.83 (0.02) / 0.30 | 0.82 (0.03) / 0.29 | 0.76 (0.03) / 0.31 |
| antiviral | rf | 0.91 (0.05) / 0.27 | 0.86 (0.06) / 0.32 | 0.79 (0.03) / 0.38 |
| antiviral | knn_tanimoto | 0.93 (0.01) / 0.17 | 0.93 (0.02) / 0.13 | 0.88 (0.02) / 0.25 |
| antiviral | trivial | 1.23 (0.08) / - | 1.20 (0.03) / - | 1.20 (0.02) / - |
| ESOL | tabpfn | 1.02 (0.07) / 0.21 | 0.98 (0.07) / 0.27 | 0.82 (0.13) / 0.36 |
| ESOL | gp_tanimoto | 1.84 (0.11) / 0.01 | 1.71 (0.03) / 0.08 | 1.56 (0.06) / 0.10 |
| ESOL | chemprop_ens | 1.49 (0.06) / -0.09 | 1.37 (0.09) / -0.00 | 0.96 (0.08) / 0.18 |
| ESOL | mlp_ens | 1.84 (0.09) / -0.00 | 1.86 (0.09) / 0.05 | 1.63 (0.12) / 0.02 |
| ESOL | rf_desc | 1.03 (0.14) / 0.14 | 0.95 (0.07) / 0.14 | 0.88 (0.06) / 0.12 |
| ESOL | rf | 1.79 (0.10) / 0.04 | 1.80 (0.06) / -0.09 | 1.66 (0.09) / 0.04 |
| ESOL | knn_tanimoto | 1.84 (0.07) / 0.15 | 1.80 (0.06) / 0.17 | 1.74 (0.17) / 0.24 |
| ESOL | trivial | 1.96 (0.00) / - | 1.99 (0.04) / - | 2.00 (0.02) / - |

Calibration (median over the 4 datasets, seed 0, draw 1): Gaussian PICP68 and the ratio mean
sigma / RMSE:

| method | PICP68 N = 30 / 120 / top | mean sigma / RMSE, N = 30 / 120 / top |
|---|---|---|
| tabpfn | 0.68 / 0.67 / 0.72 | 0.85 / 0.85 / 0.90 |
| gp_tanimoto | 0.72 / 0.74 / 0.74 | 0.98 / 1.04 / 1.04 |
| chemprop_ens | 0.06 / 0.13 / 0.25 | 0.07 / 0.18 / 0.25 |
| mlp_ens | 0.05 / 0.07 / 0.10 | 0.06 / 0.07 / 0.12 |
| rf_desc | 0.70 / 0.71 / 0.75 | 0.92 / 0.98 / 1.03 |
| rf | 0.63 / 0.66 / 0.65 | 0.90 / 0.86 / 0.94 |
| knn_tanimoto | 0.69 / 0.65 / 0.63 | 0.97 / 0.91 / 0.92 |

Reading: (i) **No new baseline beats TabPFN at small N.** At N = 30 the GP ties TabPFN on BACE
and antiviral (1.18 vs 1.15, 0.84 vs 0.85) and its sigma ranks errors about as well (rho 0.36 /
0.31 vs 0.44 / 0.35); on lipophilicity it falls back to the trivial predictor (ML-II noise-only
solution, see above) and on ESOL all Morgan-fingerprint models are poor (GP 1.78, MLP 1.94,
kNN 1.77, rf 1.69 vs descriptor-based TabPFN 0.94 / rf_desc 0.91) -- fingerprint bits carry no
size / polarity information, which dominates solubility. Over the three draws the GP is 0.04-0.11
behind TabPFN at N = 30 on the clean sets. (ii) **Chemprop is weak at small N and competitive
at the top rung**: N = 30 RMSE 1.19 / 1.26 / 1.10 / 1.48 (antiviral and BACE close to the
trivial floor), top rung 0.65 / 0.79 / 0.61 / 0.53, i.e. equal to TabPFN on antiviral (0.61),
0.02-0.03 behind on ESOL and lipophilicity and 0.12 behind on BACE. (iii) **Deep-ensemble sigmas are badly over-confident and rank
errors poorly**: median PICP68 0.05-0.25 (single draw-1 cells 0.03-0.34), median mean sigma 6-25 % of the RMSE (the members agree on
molecules far from the training set because all of them regress to the mean there), rho
-0.14-0.26 at N <= 120. The member SD is purely epistemic; without a variance head or
recalibration it is not a usable interval, only (weakly) a ranking. (iv) GP, RF and kNN sigmas
are roughly calibrated in scale (PICP68 0.63-0.74, sigma / RMSE 0.86-1.04) -- the GP closest
to nominal.

### Selective prediction with proper tie handling (`src/selective_prediction.py`, 2026-10-13; exact since 2026-10-14)

RMSE50 / RMSE80 = RMSE of the 50 % / 80 % of test molecules with the smallest sigma. Before
2026-10-13 this was computed with numpy's default argsort, so for the coarse LLM UC1 (median 29
distinct values per 150 molecules, 22 tied groups) the subset at the cut, and hence the value,
depended on the row order: over all tie orders the RMSE50 of an LLM cell spans up to 0.24
(Haiku 5.5), median 0.03, and the cut splits a tied group in 261 of 307 LLM cells. Definition
(exact, decision of the project lead, **2026-10-14**): k = floor(frac · n) slots (k = 75 / 120 for
n = 150), t = the k-th smallest sigma; the molecules with sigma < t are always selected, the m
molecules tied at sigma = t fill the remaining r = k − #{sigma < t} slots uniformly at random, so
each is selected with probability r / m and the expected MSE of the selected set is exactly
(SSE_below + (r / m) · SSE_tied) / k. **RMSE50 := sqrt of that expected MSE** -- the square root of
the expected MSE over tie orders, not the expectation of the RMSE (by Jensen the latter is <= the
former; they coincide when the cut does not split a tied group). It is deterministic and does not
depend on the row order. From 2026-10-13 to 2026-10-14 the value was instead the expected RMSE,
estimated from 200 random permutations of the tied group (`default_rng(0)`), which moved by up to
0.015 with the input row order (6 of 336 `low` LLM cells by > 0.005); the switch moved RMSE50 by
<= 0.010 and gain50 by <= 0.016 over all 1,216 cells (5 LLM / 12 baseline rows > 0.005 in RMSE50,
the 12 baseline rows being the constant-sigma `trivial` model on BACE, whose RMSE50 now equals its
RMSE, gain50 = 0), no bootstrap CI changed whether it excludes 0. `src/bootstrap_cells.py`
evaluates the same closed form per resample. The exact range over all tie orders is stored as
`RMSE50_min` / `RMSE50_max` (and for 80): the r smallest / largest squared errors of the tied group. Ties are detected on sigma rounded to 10 decimals,
because averaging UC1 over the R = 3 repeats leaves last-bit float differences (0.36666666666666664
vs 0.3666666666666667) in most cells. `results/icl/selective_prediction.csv` (replaced; 1,027
rows) covers every LLM cell of the per-molecule tables (all reasoning levels; draws as present --
on 2026-10-13 the per-molecule tables hold draw 1 only, rerun `selective_prediction.py` after
`analyze_icl.py` has picked up draws 2 / 3) and every baseline method / seed / draw; columns
`ds, model, size, RMSE, rho, RMSE80, RMSE50, gain80, gain50` (as before; gain = 1 - RMSE_F / RMSE)
plus `draw, reasoning, seed, n_ties_groups` and `family, n_mol, n_sigma_distinct, RMSE80_min,
RMSE80_max, RMSE50_min, RMSE50_max`. For tie-free sigmas (e.g. TabPFN) the values equal the old
file exactly; every old LLM value lies inside the new [min, max]. `rho` = Spearman(sigma, |err|)
with average ranks on the rounded sigma; it differs from `rho_UC1_err` of `cells.csv` (which does
not merge the float-noise ties) by <= 0.009. `report_icl.py --draws` imports `rmse_most_certain`
from this module, so `docs/RESULTS_DRAWS.md` / `draws_cells.csv` and the CSV agree exactly (checked:
254 of 254 rows); its baseline rows now include gp_tanimoto, chemprop_ens and mlp_ens.

RMSE50 (RMSE of all 150; gain50; values of the exact rule of 2026-10-14), draw 1, LLMs at reasoning `low` (R = 3 pooled: prediction and UC1
are means over the repeats), baselines seed 0. **N = 30:**

| model | lipophilicity | BACE | antiviral | ESOL |
|---|---|---|---|---|
| Sonnet 5.5 | 0.44 (0.60; 0.27) | 0.78 (0.88; 0.12) | 0.54 (0.72; 0.24) | 0.20 (0.30; 0.33) |
| Haiku 5.5 | 0.46 (0.67; 0.31) | 0.90 (1.13; 0.20) | 0.55 (0.75; 0.26) | 0.30 (0.45; 0.33) |
| GPT 6.1 Sol | 0.66 (0.78; 0.16) | 0.85 (1.02; 0.17) | 0.57 (0.74; 0.23) | 0.24 (0.45; 0.47) |
| Gemini 3.8 Flash | 0.69 (0.75; 0.09) | 0.83 (1.07; 0.23) | 0.64 (0.78; 0.18) | 0.29 (0.38; 0.25) |
| GPT-6 Luna | 0.99 (1.11; 0.10) | 0.96 (1.24; 0.22) | 0.61 (0.79; 0.24) | 0.38 (0.61; 0.37) |
| GLM 5.3 Flash | 0.91 (0.99; 0.08) | 1.05 (1.29; 0.18) | 0.63 (0.87; 0.28) | 0.46 (0.74; 0.38) |
| tabpfn | 1.12 (1.17; 0.04) | 0.78 (1.15; 0.32) | 0.66 (0.85; 0.22) | 0.64 (0.94; 0.32) |
| gp_tanimoto | 1.30 (1.33; 0.02) | 0.85 (1.18; 0.28) | 0.57 (0.84; 0.32) | 1.82 (1.78; -0.02) |
| chemprop_ens | 1.26 (1.19; -0.05) | 1.08 (1.26; 0.14) | 0.90 (1.10; 0.19) | 1.58 (1.48; -0.07) |
| mlp_ens | 1.27 (1.33; 0.04) | 1.39 (1.22; -0.14) | 0.92 (0.88; -0.05) | 1.83 (1.94; 0.06) |
| rf_desc | 1.21 (1.22; 0.01) | 0.85 (1.16; 0.27) | 0.65 (0.84; 0.22) | 0.82 (0.91; 0.10) |

**N = 120:**

| model | lipophilicity | BACE | antiviral | ESOL |
|---|---|---|---|---|
| Sonnet 5.5 | 0.47 (0.57; 0.18) | 0.65 (0.87; 0.26) | 0.55 (0.63; 0.13) | 0.21 (0.31; 0.34) |
| Haiku 5.5 | 0.49 (0.68; 0.28) | 0.65 (0.99; 0.34) | 0.47 (0.68; 0.30) | 0.29 (0.43; 0.34) |
| GPT 6.1 Sol | 0.62 (0.72; 0.14) | 0.73 (0.95; 0.23) | 0.55 (0.68; 0.20) | 0.23 (0.45; 0.49) |
| Gemini 3.8 Flash | 0.55 (0.69; 0.20) | 0.82 (1.00; 0.18) | 0.60 (0.73; 0.18) | 0.26 (0.40; 0.35) |
| GPT-6 Luna | 0.78 (0.92; 0.16) | 0.91 (1.12; 0.18) | 0.59 (0.74; 0.20) | 0.39 (0.62; 0.37) |
| GLM 5.3 Flash | 0.82 (0.91; 0.10) | 0.96 (1.19; 0.20) | 0.63 (0.82; 0.24) | 0.49 (0.77; 0.36) |
| tabpfn | 0.78 (0.94; 0.17) | 0.80 (1.02; 0.22) | 0.49 (0.79; 0.38) | 0.49 (0.78; 0.37) |
| gp_tanimoto | 0.91 (1.03; 0.12) | 0.75 (1.05; 0.28) | 0.58 (0.80; 0.28) | 1.52 (1.59; 0.04) |
| chemprop_ens | 0.80 (0.99; 0.19) | 1.05 (1.17; 0.10) | 0.74 (0.89; 0.17) | 0.79 (1.01; 0.21) |
| mlp_ens | 1.00 (1.06; 0.06) | 1.27 (1.08; -0.18) | 0.96 (0.86; -0.11) | 1.45 (1.59; 0.09) |
| rf_desc | 0.87 (0.94; 0.08) | 0.91 (1.12; 0.18) | 0.57 (0.79; 0.28) | 0.79 (0.88; 0.10) |

Reading: at N = 30 the strong LLMs' most-certain half is better than any classical model's on
lipophilicity (Sonnet 5.5 0.44, Haiku 5.5 0.46 vs best baseline 1.12, TabPFN) and ESOL (0.20 /
0.30 vs 0.64), and roughly equal on antiviral (0.54 / 0.55 vs GP 0.57, rf_desc 0.65) and BACE
(Sonnet 0.78 = TabPFN 0.78, Haiku 0.90 vs GP / rf_desc 0.85). How much selecting helps (gain50)
depends on the dataset: on lipophilicity the Sonnet / Haiku sigma selects much better than any
baseline sigma (0.27 / 0.31 vs 0.01-0.04 for TabPFN / GP / rf_desc), on ESOL somewhat better
(LLMs 0.25-0.47 vs 0.10-0.32), on antiviral about equally (0.18-0.28 vs 0.22-0.32) and on BACE
worse (0.12-0.23 vs 0.27-0.32); so on BACE / antiviral the LLM advantage at N = 30 is accuracy,
not error ranking. The deep ensembles gain nothing or lose (gain50 down to -0.14 at N = 30 and
-0.18 at N = 120). At N = 120 TabPFN's selected half catches up on antiviral (0.49 vs Haiku 0.47,
Sonnet 0.55), the GP's on BACE comes within 0.10 of Sonnet / Haiku (0.75 vs 0.65), and the LLM
advantage on lipophilicity (0.47 / 0.49 vs >= 0.78) and ESOL (recall, section 11) stays.

## 8. Order of work

1. ~~Corpus search tiers 2-3 -> final test sets~~ done 2026-09-09.
2. Effort-dosability probe of the new registry: `PROBE_DRYRUN=1 julia src/julia/Probe_Models.jl`,
   then `PROBE_REPEATS=2 julia src/julia/Probe_Models.jl` (2 repeats recommended for the panel;
   hard ceiling `PROBE_BUDGET`, default $3, reserving 2x the worst case per call in flight) ->
   `results/meta/effort_probe_<date>.jsonl` (per call, as it arrives) and `.json` (summary);
   enter `reasoning_levels` in the registry by hand.
3. `ZS_PROBE=1 julia src/julia/Run_ZeroShot_TestSet.jl`, then the zero-shot check at `low` (4200 calls).
4. MANDATORY: `ICL_PROBE=1 ICL_SIZES=max julia src/julia/Run_ICL.jl` -> measured output tokens / cache hits
   (Kimi K3!) / billed cost per call -> refined cost estimate -> go / no-go.
   **Steps 2 and 4 done 2026-10-08 ($5.28, section 9).** opus55 and dsv41flash resolved
   2026-10-09 by the panel decision (both dropped, section 10); step 3 (zero-shot check, six
   models) started 2026-10-09. Still open: `reasoning_levels` to enter by hand.
5. Pilot: one dataset, sizes 30 and 1000, panel, R = 3 (~$100-150 by the 2026-10-08 assumptions;
   ~$17 for lipophilicity with the six-model panel and the measured estimate of section 10).
6. Main run per dataset; ladder afterwards.

## 9. Probe results 2026-10-08

First paid calls of v2 (authorised by the project lead for these two probes only; ceilings $25
effort probe, $15 ICL probe). Raw outputs in `results/meta/`; merged by `src/summarize_probes.py`
-> `results/meta/probe_summary.csv` (per model the latest file with an answered `low` call).

**Spend.** Effort probe $2.59 (batch 1, the 18 cheaper models: $1.943 of a $10 share; batch 2,
sonnet55 / opus55 / kimik3 / opus5 / gpt6astra: $0.609 of $15; retry of gpt61sol + the two
DeepSeek tags: $0.039 of $2). The probe was split into two batches because the dry run's sum of
worst cases was $96.50 for a $25 ceiling (2x reservation of 32k output tokens per call), so a single
run could have starved the expensive models near the ceiling. ICL probe $2.69 (all 23 tags: $2.622 of
$15; retry of gpt61sol + opus55: $0.067 of $3). **Total $5.28** (sum of the billed `usage.cost` of
every call). The shared library counter (`LLMUsage.json`) went from $7,598.03 to $7,603.17
(+$5.14). The $0.14 gap is probably lost updates while up to three Julia processes wrote that file
concurrently (not verified).

Files: `effort_probe_2026-10-08.json(l)` (batch 1), `_2` (batch 2), `_3` (retry);
`icl_probe_2026-10-08.json(l)` (all tags), `_2` (retry). The ICL probe used lipophilicity,
N = 1000, molecule CHEMBL1951022 (truth 2.8), effort `low`, max_tokens 32000, two identical calls
per model. The effort probe used the zero-shot prompt on the same molecule, 5 levels x 2 repeats.

**Code changes made for the probes.** `Run_ICL.jl` ICL_PROBE block rewritten: models run
concurrently, calls within a model run sequentially; it reserves budget like Probe_Models.jl
(worst case 2 x (3 x chars/4 prompt at max(price_in, price_cache_write) + max_tokens at price_out))
and stops at ICL_BUDGET. It writes `icl_probe_<date>[_k].jsonl` per call plus a `.json`, and
`ICL_PROBE_PLAN=1` shows the plan only. `Common.jl::resolve_tags` handles `all` / `panel` for
ICL_MODELS / PROBE_MODELS / ZS_MODELS. `ask_gpt_full` now also returns `cache_write_tokens`.
`estimate_icl_cost.py --measured results/meta/probe_summary.csv` takes output tokens, cache share
and tokens per record per model.

Ranked by the projected main run (4 datasets x 6 sizes x 150 molecules, `low`, R = 1 / R = 3,
USD; measured output tokens, call-2 cache share and tokenizer factor of that model; cache-write
premiums not modelled). "rsn tok @low (ZS)" = median reasoning tokens at `low` in the effort probe
(zero-shot prompt). "ICL out / rsn" = mean completion / reasoning tokens of the ICL-probe calls.

| tag | panel | effort verdict | suggested levels | rsn tok @low (ZS) | ICL out / rsn tok | cache share call 2 | $ call 1 / call 2 | parse | finish | served by | proj. main R=1 / R=3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt6luna |  | monotone | none/minimal/low/medium/high | 851 | 500 / 468 | 1.00 | 0.0032 / 0.0004 | 2/2 | stop/stop | OpenAI | 1 / 3 |
| haiku55 | yes | non-monotone (ties at 0) | none/low/high | 0 | 252 / 102 | 1.00 | 0.0087 / 0.0008 | 2/2 | stop/stop | Anthropic | 2 / 5 |
| glm53flash |  | non-monotone | minimal/low/medium/high | 33 | 132 / 104 | 1.00 | 0.0072 / 0.0015 | 2/2 | stop/stop | Z.AI | 3 / 8 |
| gem38flash | yes | non-monotone | minimal/low/medium/high | 314 | 763 / 733 | 1.00 | 0.0030 / 0.0048 | 2/2 | stop/stop | Google AI Studio | 9 / 27 |
| gpt61sol | yes | monotone (low not answered) | minimal/medium/high (+low) | -- | 324 / 292 | 1.00 | 0.0624 / 0.0041 | 2/2 | stop/stop | OpenAI | 12 / 37 |
| sol |  | monotone | none/minimal/low/medium/high | 1253 | 511 / 479 | 1.00 | 0.0635 / 0.0074 | 2/2 | stop/stop | OpenAI | 19 / 57 |
| glm53 |  | non-monotone | minimal/low/medium/high | 32 | 131 / 104 | 1.00 | 0.0674 / 0.0129 | 2/2 | stop/stop | Z.AI | 23 / 68 |
| terra |  | monotone on none<low<high | none/minimal/low/high | 1235 | 628 / 596 | 1.00 | 0.0652 / 0.0081 | 2/2 | stop/stop | OpenAI | 23 / 70 |
| sonnet55 | yes | flat (0 reasoning tokens at every level) | low | 0 | 313 / 0 | 1.00 | 0.1751 / 0.0103 | 2/2 | stop/stop | Anthropic | 30 / 90 |
| gem36flash |  | monotone | minimal/low/medium/high | 734 | 1052 / 1022 | -- (call 2: 429) | 0.0050 / -- | 1/1 | stop | Google | 32 / 96 |
| kimik3 | yes | monotone on none<low<high | none/minimal/low/medium/high | 338 | 732 / 692 | 1.00 | 0.1582 / 0.0249 | 2/2 | stop/stop | Moonshot AI | 69 / 206 |
| opus5 |  | non-monotone (0 below medium) | none/low/medium/high | 0 | 35 / 0 | 1.00 | 0.4299 / 0.0367 | 2/2 | stop/stop | Anthropic | 72 / 215 |
| qwen3827b |  | non-monotone | none/minimal/low/medium | 1486 | 7452 / 4010 | 1.00 | 0.0526 / 0.0134 | 2/2 | stop/stop | Alibaba | 75 / 226 |
| qwen38big |  | flat | low | 662 | 2542 / 2070 | 1.00 | 0.1340 / 0.0266 | 2/2 | stop/stop | Alibaba | 78 / 233 |
| gpt6astra |  | monotone | minimal/low/medium/high | 259 | 350 / 318 | 1.00 | 0.3130 / 0.0332 | 2/2 | stop/stop | OpenAI | 80 / 239 |
| grok47 |  | monotone on minimal<low<high | minimal/low/medium | 2027 | 2902 / 2878 | 1.00 | 0.1211 / 0.0395 | 2/2 | stop/stop | xAI | 102 / 307 |
| grok45 |  | monotone | minimal/low/medium/high | 412 | 338 / 314 | **0.00** | 0.1016 / 0.1016 | 2/2 | stop/stop | xAI | 140 / 419 |
| hy4preview |  | on/off only (32k cap at every level) | none/low | 32000 | 30552 / 30538 | 1.00 | 0.0988 / 0.0738 | 1/2 | stop/length | Tencent | (252 / 756) |
| mistrall4 |  | on/off only | none/low | 13969 | 32000 / 20594 | **0.00** | 0.0989 / 0.0989 | **0/2** | length/length | Mistral | (283 / 850) |
| step5preview |  | monotone (completion tokens; no reasoning reported) | minimal/low/medium/high | 0 | 32000 / 0 | 1.00 | 0.1323 / 0.0888 | **0/2** | length/length | StepFun | (317 / 952) |
| opus55 | yes | on/off only (0 at minimal/low, ~100 at medium/high) | minimal/low | 0 | **refused** | -- | 0 / 0 | **0/4** | content_filter x4 | Anthropic | -- |
| dsv41flash | yes | -- (404) | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| dsv4pro |  | -- (404) | -- | -- | -- | -- | -- | -- | -- | -- | -- |

Projections in parentheses are lower bounds, because the output hit the 32k cap and no usable
answer came back. Suggested `reasoning_levels` are the probe's output and are **not** entered in the
registry (still `null`; entering them is the project lead's call, see the notes on ties below).

**Panel, measured** (`estimate_icl_cost.py --measured results/meta/probe_summary.csv`): main run
haiku55 $2 / sonnet55 $30 / gpt61sol $12 / gem38flash $9 / kimik3 $69 at R = 1 (= **$122**;
R = 3: $5 / $90 / $37 / $27 / $206 = **$365**). The 2026-10-08 assumptions (1200 output tokens,
80 % cache) overestimated every measured panel model. They emit 250-760 completion tokens at
`low` on the ICL prompt, and every model except grok45 and mistrall4 served >= 99.5 % of the
prompt from cache on the second call. That includes Kimi K3 (48,640 / 48,743), so the
"Kimi K3 cache unmeasured" risk of section 6 is resolved. opus55 and dsv41flash have no measurement.
With the old assumptions for those two the panel total is $286 (R = 1) / $859 (R = 3). Ladder
(other levels, assumed output tokens, measured cache / tokenizer): $830 at R = 1 for the panel.
CSVs: `results/icl_cost_estimate_2026-10-08_measured_R1.csv`, `_R3.csv` (all measured tags).
(This paragraph is the seven-model panel of 2026-10-08; for the six-model panel of 2026-10-09
see section 10.)

**Models that fail / must not go into the main run as configured:**

1. **opus55 (panel): refuses the ICL prompt.** All 4 ICL calls (two in the main probe, two in the
   retry) came back with HTTP 200, `finish_reason = content_filter`, 0 prompt and 0 completion
   tokens, $0. On the short zero-shot prompt it answers (7 of the 8 calls at minimal..high;
   one content_filter at `high`; `none` is rejected with HTTP 400). The likely trigger is the 1000-record SMILES block, but this is not verified. Opus 5
   and Haiku / Sonnet 5.5 answer the identical request. Decision needed: replace (opus5 works:
   35 output tokens at `low`, $72 / $215 at R = 1 / 3) or investigate further (smaller N, a
   different dataset).
2. **dsv41flash (panel) and dsv4pro: unreachable under the current account settings.** HTTP 404
   at routing: "Filter by Guardrails removed deepseek (Paid model training violation (account
   settings) ...)". The OpenRouter account's privacy guardrail excludes providers that may train
   on paid prompts, and DeepSeek's own endpoint is one of them. Not an outage, so a retry cannot
   fix it (both retries failed the same way, $0). Options for the project lead: allow such
   providers in the account's privacy settings (an account decision), pin a non-training
   third-party host (quantisation and price differ), or swap the model.
3. **mistrall4, hy4preview, step5preview: runaway output.** 14-17k reasoning tokens at every level from `minimal` on
   (Mistral) or at every level (Hy4, truncated at 32k on 7 of 10 effort calls). In the ICL probe
   all three hit the 32k cap (step5preview and mistrall4 2/2, hy4preview 1/2), and only one call
   in six parsed. 5-15 min per call. Not usable for the ICL protocol at max_tokens 32000.
4. **grok45: no prompt-cache hit** (128 of 49,893 tokens cached on call 2; the only unpinned
   entry, default routing to xAI), so it is the most expensive working reserve ($140 / $419).
   mistrall4 also shows no cache.
5. **gem36flash:** call 2 of the ICL probe got HTTP 429 (rate limit), so its cache share is
   unmeasured (reserve, not retried).
6. **gpt6astra:** works (monotone, 350 output tokens, 100 % cache), but it is the probe-only
   reserve ($80 / $239).

**Dosability (effort probe):** `none` is rejected (HTTP 400, "Reasoning is mandatory") by
gpt61sol, gem38flash, sonnet55, opus55, gpt6astra, gem36flash, grok45/47, glm53, glm53flash,
qwen38big and step5preview. The Anthropic 5.5 models barely reason at the levels used here:
Haiku 0 reasoning tokens from `none` to `medium` and ~175 at `high`; Sonnet 0 at every level
(answers with ~230 visible tokens; flat); Opus 5.5 0 at minimal/low and ~100 at medium/high.
Several "non-monotone" verdicts (haiku55, opus5) come from ties at 0 tokens, not from inversions.
Monotone ladders: gpt6luna, sol, gpt6astra, gem36flash, grok45 (full), gpt61sol (minimal 374 <
medium 701 < high 2,690 median; `low` never answered in the effort probe, see below), terra and
kimik3 (on none<low<high). Flat: sonnet55, qwen38big. GLM 5.3 / 5.3 Flash emit only 30-120
tokens at minimal..medium and are effectively flat.

**Oddities.**

- **Flex was applied and nothing fell back.** Every billed / list cost ratio is 1.00
  (gem38flash 1.02-1.03, gem36flash 1.02; Qwen call 2 0.87, i.e. Alibaba bills cache reads below
  the listed price). So the flex prices (OpenAI incl. gpt61sol, Google AI Studio) were billed. No
  call was served by a provider other than the pinned one, and every served model id matches the
  requested endpoint.
- **gpt61sol flex is intermittent.** The `openai/flex` endpoint reported status -5 (0 % uptime
  over 30 min) at 19:00 and again at 20:00 UTC. Eight effort-probe calls in the first run and
  the retry got 502 "Flex processing is temporarily unavailable", including all four `low`
  calls, so its effort-probe `low` is unmeasured. Between failures it answered (both `high`
  calls in run 1; minimal, medium and high in the retry), and the ICL retry succeeded on both
  calls: 307 / 341 completion tokens, 275 / 309 reasoning tokens, 100 % cache on call 2,
  $0.062 / $0.004, 12-13 s. Because allow_fallbacks is false, a flex outage fails the call
  instead of billing the 2x standard endpoint. The main-run runner retries 5x with 10 s pauses
  per call; long outages will leave gaps, which the resumable runner can fill later.
- **OpenAI flex reports and bills a cache write on call 1** (48,696 `cache_write_tokens` at the
  1.25x write price). Gemini reports the call-1 prompt as both cached and written. Neither premium
  is modelled in the estimate (it occurs once per cell and cache lifetime).
- **Claude tokenizers count 1.41x o200k** on this prompt (68,715 vs 48,685 tokens). BACE at
  N = 1000 is 64,371 o200k tokens, so about 91k on Claude. That is still below Haiku 5.5's 100k
  long-context surcharge, but with only about 9 % margin. Check the measured prompt tokens in the
  pilot.
- **Effort ignored in different directions:** Haiku's ICL call 1 emitted 205 reasoning tokens at
  `low`, although it emitted 0 on the zero-shot prompt at every level below `high`.

## 10. Panel decision, prompt revision and MiMo probe 2026-10-09

**Decisions (project owner, 2026-10-09).** Panel = haiku55, sonnet55, gpt6luna (flex), gpt61sol
(flex), gem38flash (flex), glm53flash (table and prices in section 4). Temperature 1.0 (provider
default, omitted from the body) and R = 1 first, R = 3 later, as decided 2026-10-08. ICL prompt
revised (section 4, "Prompt revision 2026-10-09"; parity Julia <-> Python IDENTICAL for all four
datasets); zero-shot prompt unchanged (section 3).

**Dropped, and why.**

1. **opus55 -- refuses the ICL prompt from N >= 250.** Size diagnostic of 2026-10-08 evening
   (`results/meta/diagnostics/icl_probe_2026-10-08_3.json(l)` / `_4`, lipophilicity, CHEMBL1951022,
   `low`, max_tokens 4000): N = 30 answered 2/2 (269 / 346 completion tokens, parsed, $0.018 /
   $0.009); N = 250 refused 2/2 with `finish_reason = content_filter` (call 1: 17,587 prompt and
   269 completion tokens billed, $0.093; call 2: 0 tokens, $0). At N = 1000 all four calls of
   2026-10-08 were refused (section 9). Diagnostic spend $0.12. A model that refuses from the
   third rung on cannot cover the ladder.
2. **dsv41flash -- blocked by the OpenRouter account's privacy setting** (HTTP 404, the
   guardrail excludes providers that train on paid prompts, DeepSeek's own endpoint among them;
   section 9). Changing the account setting was not chosen.
3. **kimik3 -- too expensive:** $69 / $206 projected main run at R = 1 / R = 3 (measured), more
   than the other six panel models together.
4. **MiMo V2.6 (mimo26pro, mimo26flash; added to the registry 2026-10-09 on request) --
   reasoning is either off or unbounded.** OpenRouter lists `reasoning` but no
   `reasoning_effort` for either endpoint. Probe files: `results/meta/effort_probe_2026-10-09.jsonl`
   (8 records) and `results/meta/icl_probe_2026-10-09.jsonl` (2 records); neither run wrote its
   `.json` summary, i.e. both ended before completion (presumably stopped by hand; not
   verified). Contents (lipophilicity CHEMBL1951022, truth 2.8, max_tokens 32000, pinned to
   Xiaomi):

   | probe | tag | level | result | completion / reasoning tok | latency | cost |
   |---|---|---|---|---|---|---|
   | effort (ZS prompt) | mimo26pro | none | 2/2 answered: 2.74, 4.08 | 5 / 0 | 3-5 s | $0.00013 total |
   | effort (ZS prompt) | mimo26pro | minimal | 1/1 no answer: `Connection closed after 900 seconds` (1810 s wall, presumably one retry) | -- | -- | $0 |
   | effort (ZS prompt) | mimo26flash | none | 2/2 answered: 3.76, 3.59 | 5 / 0 | 4-16 s | $0.00006 total |
   | effort (ZS prompt) | mimo26flash | minimal | 2/2 truncated (`finish = length`, empty text) | 32,000 / 31,999 | 621-772 s | $0.009 each |
   | effort (ZS prompt) | mimo26flash | low | 1/1 truncated (`length`, empty text) | 32,000 / 32,000 | 540 s | $0.009 |
   | ICL (N = 1000, `low`) | mimo26flash | low | truncated, no parse | 32,000 / 32,001 | 782 s | $0.016 |
   | ICL (N = 1000, `low`) | mimo26pro | low | parsed: 2.9 +/- 0.5 | 4,284 / 3,697 | 110 s | $0.004 |

   The Pro model's single ICL call worked (and already reported 48,640 cached of 48,656 prompt
   tokens on call 1, cause not verified), but its `minimal` zero-shot call never returned and
   `medium` / `high` were not reached; the Flash model emits no reasoning at `none` and runs into the
   32k cap at every level tested above it (`minimal`, `low`). No dosable effort ladder and no controllable cost at
   `low`. Spend: effort probe $0.027, ICL probe $0.029, **total $0.056**.

**Cost of the six-model panel (measured mode).**

    python src/estimate_icl_cost.py --measured results/meta/probe_summary.csv \
        --models haiku55,sonnet55,gpt61sol,gem38flash,gpt6luna,glm53flash --repeats 1 \
        --csv results/icl_cost_estimate_2026-10-09_panel6_R1.csv      (and --repeats 3 ... _R3.csv)

All six models have measured parameters (ICL output tokens, call-2 cache share 1.00, tokenizer
factor 0.98-1.41). USD, rounded:

| | haiku55 | sonnet55 | gpt61sol | gem38flash | gpt6luna | glm53flash | total |
|---|---|---|---|---|---|---|---|
| main run, `low`, R = 1 (21,600 calls) | 2 | 30 | 12 | 9 | 1 | 3 | **57** |
| main run, `low`, R = 3 (64,800 calls) | 5 | 90 | 37 | 27 | 3 | 8 | **170** |
| reasoning ladder (none / minimal / medium / high at min & max size), R = 1 | 8 | 141 | 66 | 27 | 4 | 10 | **256** |
| reasoning ladder, R = 3 | 24 | 424 | 197 | 82 | 11 | 30 | **768** |

Main run + ladder: $313 at R = 1, $938 at R = 3. Sonnet 5.5 is about half of every total. The
ladder rows use the assumed output tokens per level (none 150 / minimal 400 / medium 2500 /
high 6000), not measurements, and include `none` for the four models that reject it, so they are
an upper bound of the ladder as specified. The main-run figures were measured with the
pre-revision prompt (2026-10-08); the revision changes the prompt length negligibly. Seven-model
panel of 2026-10-08 for comparison: $122 / $365 (its five measured models) or $286 / $859 with assumptions
for opus55 and dsv41flash (section 9).

## 11. Main run R = 1 -- results (2026-10-09)

Run: 6 panel models x 4 datasets x 6 context sizes, 150 test molecules, 1 call per molecule,
reasoning `low`, temperature 1.0; 144 cells, 21,600 planned calls, $55.81 (projection was $57),
6.5 h wall-clock at 8 concurrent requests, parse rate 100 % of stored records, no truncation,
gap-fill rerun afterwards (`results/icl/run_R1_2026-10-09.log`, `run_R1_gapfill_2026-10-09.log`).
Tables: `docs/RESULTS_ICL_R1.md` (generated by `src/report_icl.py` from `results/icl/cells.csv`).
Independent review of the analysis (Opus 5.5, 2026-10-09): all cell metrics reproduced from the
raw JSONL to 1e-6; defects fixed afterwards in `src/analyze_icl.py` (never-recalled subset must
exclude unscorable < 3-figure labels: ESOL 54 not 91 molecules; PICP68 at z = 1.0 with a 1e-9
tolerance; sigma floor 0.01 with a clip counter; duplicate (mol_id, repeat) records dropped;
coverage = answered molecules / 150 reported next to the parse rate).

Known gap: during the main run the OpenAI endpoint (GPT 6.1 Sol, GPT-6 Luna) rejected a few
individual antiviral prompts per cell (2-7 of 150) and two ESOL prompts for Luna, citing its own
content policy; the review found the gaps stochastic (different molecules at different sizes),
and the gap-fill rerun (same prompts, no rephrasing, $0.23) recovered all but one molecule in two
cells (antiviral N = 120 for Sol and for Luna: 149 of 150). Coverage is reported per cell; final
totals 21,598 calls, $56.03.

Findings (qualified by the review):
1. **Clean datasets show real learning curves.** Every model improves from N = 30 to the top
   rung on lipophilicity, BACE and antiviral (8-34 % RMSE): Sonnet 5.5 0.60 -> 0.50 / 0.90 -> 0.68 /
   0.72 -> 0.53. At N = 30 the stronger LLMs are well ahead of TabPFN (1.17 / 1.15 / 0.85) on all
   three sets; GLM 5.3 Flash and GPT-6 Luna are not ahead on BACE (1.30 / 1.26) and GLM not on
   antiviral (0.90). At the top rung, paired on identical molecules (Wilcoxon): Sonnet 5.5, GPT 6.1
   Sol and Gemini 3.8 Flash beat TabPFN on lipophilicity (-0.10 to -0.12 RMSE, p <= 0.008); BACE is
   a tie for Sonnet/Sol (0.68-0.69 vs 0.67) and a loss for the rest; on antiviral the lower LLM RMSE
   (0.52-0.53 vs 0.61) is driven by a few large TabPFN errors (per-molecule medians favour TabPFN,
   p ~ 0.9). The N = 60 point is worse than N = 30 for every LLM on antiviral while TabPFN barely
   moves: the LLMs are sensitive to the particular 30 extra examples, and with R = 1 and one
   nested context draw per size the curve points carry draw noise.
2. **ESOL is flat.** RMSE changes by <= 0.06 from N = 30 to 976 (Sonnet 0.33 -> 0.27, Gemini
   0.41 -> 0.38, Sol 0.47 -> 0.45) and Spearman is 0.97-0.98 at N = 30 already, while TabPFN
   improves 0.95 -> 0.51 on the same cells. The predictions themselves barely move (Spearman
   between the N = 30 and N = 976 prediction vectors 0.94-0.99 vs 0.38-0.90 on the clean sets;
   Sonnet repeats the identical value for 38 % of ESOL molecules). Flatness also holds for the
   zero-shot-clean GLM 5.3 Flash and on the never-recalled scorable subset (54 molecules), so it
   is a "prior dominates" signature (recall plus general solubility knowledge), not a pure
   memorisation signature. What separates recalled (59) from never-recalled molecules is the error
   level: 2-3x lower RMSE on recalled molecules for Sonnet (0.16 vs 0.39), Sol (0.26 vs 0.67) and
   Gemini (0.20 vs 0.54). Sonnet names a retrieval source in 96 % of its ESOL answers (AqSolDB in
   79 %) and is 4x more accurate when it does; on lipophilicity it claims recall in 21 % of calls
   with no accuracy gain (confabulated recall).
3. **Calibration of the self-reported sigma.** Clean sets: PICP68 0.40-0.81, per-dataset medians
   0.57-0.63 (over-confident), with exceptions near nominal (Haiku 5.5 on lipophilicity 0.76-0.81,
   GPT 6.1 Sol 0.81 at the top rung on lipophilicity and antiviral); rho(UC1, |err|) 0.10-0.45.
   ESOL: PICP68 0.68-0.88 (at or above nominal, i.e. under-confident for the heavy recallers),
   PICP90 0.85-0.95, rho 0.29-0.56 -- a mixture of recalled molecules with PICP68 0.95-1.0 and
   never-recalled ones at 0.55-0.78. This is the v1 lesson in the data: on a memorised benchmark
   the uncertainty looks calibrated for the wrong reason.
4. **Reasoning tokens at `low`:** Sonnet 5.5 0 on every call, Haiku 5.5 0 on 97 %, GPT 6.1 Sol
   medians 82-246, GPT-6 Luna 410-510, GLM ~80, Gemini bimodal (33 % of calls report 0 and write
   the reasoning as visible text instead). The Anthropic models still write ~350 tokens of
   visible reasoning, so 0 reasoning tokens is not "no reasoning".
5. **Uncertainty values** are coarse (9-37 distinct values per model, mode 14-40 % of calls),
   never zero, never absurd (0.08-1.7, below 3x the label SD); Gemini repeats the JSON block in
   2.4 % of calls (parsed identically).
6. **Zero-shot scoring caveat** (contamination check, section 3): Sonnet 5.5 answered ESOL in
   mol/L instead of log units for 38 of 150 molecules (Haiku 28); 9 of Sonnet's become exact
   after log10, so its ESOL hit3 of 31 % is a lower bound (verdict unchanged). Lipophilicity's
   clean verdicts rest on the 81 scorable 3-figure labels.

Open for the next step: R = 3 (adds repeats 1-2 only, ~$110 at the measured prices) gives UC3
and averages the draw noise; the reasoning ladder (section 6) is still undecided.

## 12. Main run extended to R = 3 (2026-10-12)

Repeats 1 and 2 added by re-running `Run_ICL.jl` with `ICL_REPEATS=3` (resumable by (mol_id,
repeat)); three detached processes by model group (sonnet | gpt61sol + gem38flash | haiku55 +
gpt6luna + glm53flash), 10 concurrent requests each, plus a gap-fill rerun. Cost of the extension
$105.68 + $0.26 gap-fill; the main experiment now totals 64,795 calls / $161.98 (every cell has
>= 448 of 450 calls, every molecule answered). Tables: `docs/RESULTS_ICL_R3.md`
(`src/report_icl.py --reasoning low --out docs/RESULTS_ICL_R3.md`; predictions and UC1 are means
over the parsed repeats, UC3 = variance of the per-repeat predictions).

Top-rung summary (RMSE N = 30 -> top; PICP68; rho(UC1, |err|); rho(UC3, |err|)):

| dataset | Sonnet 5.5 | Haiku 5.5 | GPT 6.1 Sol | Gemini 3.8 Flash | GPT-6 Luna | GLM 5.3 Flash |
|---|---|---|---|---|---|---|
| lipophilicity | 0.60 -> 0.48; 0.69; 0.21 / 0.19 | 0.67 -> 0.55; 0.79; 0.12 / 0.08 | 0.78 -> 0.51; 0.81; 0.31 / 0.21 | 0.75 -> 0.50; 0.69; 0.26 / 0.12 | 1.11 -> 0.71; 0.59; 0.27 / 0.18 | 0.99 -> 0.75; 0.60; 0.23 / 0.20 |
| BACE | 0.88 -> 0.64; 0.65; 0.29 / 0.15 | 1.13 -> 0.81; 0.61; 0.39 / 0.16 | 1.02 -> 0.69; 0.75; 0.31 / 0.22 | 1.07 -> 0.75; 0.59; 0.24 / 0.17 | 1.24 -> 0.79; 0.52; 0.25 / 0.26 | 1.29 -> 0.86; 0.65; 0.33 / 0.30 |
| antiviral | 0.72 -> 0.52; 0.72; 0.37 / 0.29 | 0.75 -> 0.57; 0.70; 0.31 / 0.25 | 0.74 -> 0.53; 0.81; 0.34 / 0.19 | 0.78 -> 0.58; 0.67; 0.30 / 0.32 | 0.79 -> 0.67; 0.63; 0.27 / 0.14 | 0.87 -> 0.80; 0.64; 0.35 / 0.27 |
| ESOL | 0.30 -> 0.29; 0.87; 0.40 / 0.66 | 0.45 -> 0.44; 0.81; 0.42 / 0.51 | 0.45 -> 0.40; 0.86; 0.55 / 0.56 | 0.38 -> 0.34; 0.81; 0.43 / 0.45 | 0.61 -> 0.56; 0.70; 0.54 / 0.50 | 0.74 -> 0.73; 0.75; 0.28 / 0.12 |

Reading: averaging three repeats lowers the RMSE by 0.01-0.04 against R = 1 (draw noise), the
learning-curve and calibration picture of section 11 is unchanged. UC3 (repeat spread) ranks the
errors no better than the self-reported UC1 on the clean sets (rho 0.08-0.32 vs 0.12-0.39) and
is clearly better only on ESOL for Sonnet (0.66 vs 0.40), i.e. where the model is unsure which
value it "remembers". The full calibration suite (PIT/ECE, ENCE, recalibration) is the next
analysis step (`src/v1reuse/uq_methods_eval.py`).

## 13. Reasoning ladder R = 1 -- results (2026-10-12)

Efforts none/minimal/low/medium/high at the smallest (30) and largest context (1000 / 844 / 976),
`low` reused from the main run, 3 detached processes by model group with 12 concurrent requests
each; 192 new cells, 27,900 calls, $135.09 (Sonnet $36.57, Haiku/Luna/GLM $8.89, Sol/Gemini $89.63).
`none` is rejected (HTTP 400 "Reasoning is mandatory") by Sonnet 5.5, GPT 6.1 Sol, Gemini 3.8 Flash
and GLM 5.3 Flash. Tables `docs/RESULTS_LADDER_R1.md`, rows `results/icl/ladder_cells.csv`
(`src/report_icl.py --ladder`). R = 3 extension running (section 14 when done).

Findings (top rung): (1) dosability is model-specific -- Sonnet 5.5 emits 0 reasoning tokens at
every level and its metrics are identical across levels; GLM reacts only at `high` (340-700
tokens), Haiku from `medium` (180-480), Sol scales 90 -> 1,900, Luna 380 -> 2,100, Gemini 440 ->
30,700 tokens ($7-8 per 150 calls at `high`). (2) Accuracy gains from more reasoning are small and
inconsistent: Haiku -0.02 to -0.05 RMSE, Sol only on antiviral (0.53 -> 0.48), Gemini BACE
0.75 -> 0.66 / antiviral 0.58 -> 0.43 / ESOL 0.34 -> 0.31 but lipophilicity 0.50 -> 0.56, GLM ESOL
0.73 -> 0.60; `none` is clearly worse for Luna (ESOL 0.83 vs 0.56). (3) Uncertainty quality does
not improve with reasoning: PICP68 and rho(UC1,|err|) are flat across levels, Gemini at `high`
is more over-confident (BACE PICP68 0.45). (4) `low` is the sensible default for this panel.

## 14. Reasoning ladder extended to R = 3 (2026-10-13)

Repeats 1-2 added for all ladder cells (sonnet $65.14, haiku/luna/glm $16.87, sol $21.06 + the
combined sol/gem process before it was split $34.73, gem $119.90, gap-fill $0.39 = $258.09, from the
run logs; ladder total $394.97 = sum of `cost_usd` over the 160 non-`low` cells of `results/icl/cells.csv`,
the value to quote; 160 cells, 71,999 calls, every cell >= 449 of 450). *Correction 2026-10-14: the
ladder total read "$393.18" (log-based sum) until 2026-10-14; the stored calls in cells.csv sum to
$394.97.* The combined Sol/Gemini R = 3
process was stopped and relaunched as two processes at 21:58 (resumable; in-flight calls lost only)
so that Gemini's `high` cells (up to 45 min / $10 per 300 calls) did not block Sol. Tables
`docs/RESULTS_LADDER_R1.md` now hold the R = 3 numbers (`src/report_icl.py --ladder`).

Findings unchanged from section 13 with smaller noise: Sonnet 5.5 flat across efforts (effort
ignored); Haiku gains from `medium` on (BACE 0.81 -> 0.74, ESOL 0.44 -> 0.39); Sol `high` helps only
on antiviral (0.53 -> 0.47) and ESOL (0.40 -> 0.36); Gemini `high` (17-31k tokens) helps on BACE
(0.75 -> 0.67), antiviral (0.58 -> 0.49) and ESOL (0.34 -> 0.32) but hurts lipophilicity (0.50 ->
0.56) and makes it more over-confident (PICP68 0.49-0.60); GLM `high` helps (ESOL 0.73 -> 0.55); Luna
`none` is clearly worse. PICP68 and rho(UC1,|err|) do not improve with reasoning for any model.

## 15. Context draws at N = 30 / 60 / 120 (2026-10-13)

Why: the main experiment uses one nested training order per dataset, so the small-N points of the
learning curve carry draw noise (section 11: every LLM is worse at N = 60 than at N = 30 on
antiviral, TabPFN barely moves). Two more independent nested context draws per dataset give three
draws per (dataset, model, N) at N = 30 / 60 / 120; the larger rungs keep draw 1 only.

**Splits** (`python src/make_splits.py --draws 3`). Draw 1 = `data/splits/<ds>_v2.json`, never
rewritten (the script regenerates it and asserts field-by-field equality; test sets asserted
byte-identical). Draw d = 2, 3: `data/splits/<ds>_v2_draw<d>.json` with the same `test_ids`,
`sizes`, `n_train_pool` and the same deduplicated pool (same molecule set, so the InChIKey-14 dedup
against the test set holds exactly as for draw 1; re-checked), `train_order_ids` = an independent
permutation of draw 1's pool, `numpy.random.default_rng(seed_draw).permutation(len(pool))` with
`seed_draw = int.from_bytes(sha256("20260907:draw<d>").digest()[:4], "big")` = 2959721235 (draw 2)
and 346829934 (draw 3); extra fields `draw`, `seed_draw`, `draw_rule`. The first N of a draw's
order are its in-context set (nested across 30 / 60 / 120). The default `--require` of
`make_splits.py` is now `all` (the final rule). Overlap of the first 120 molecules with draw 1
(`results/meta/context_draw_overlap.csv`; expected 120^2 / pool):

| dataset | pool | draw 2 vs 1 | draw 3 vs 1 | draw 3 vs 2 | expected | first 30: 2 vs 1 / 3 vs 1 |
|---|---|---|---|---|---|---|
| lipophilicity | 4042 | 1 | 4 | 5 | 3.6 | 0 / 0 |
| bace | 1355 | 7 | 9 | 10 | 10.6 | 0 / 1 |
| antiviral | 844 | 20 | 17 | 18 | 17.1 | 1 / 0 |
| esol | 976 | 18 | 19 | 12 | 14.8 | 0 / 1 |

**Pipeline.** Python `common.load_split(key, draw=1)` / `split_path`, `icl_common.cell_data(key,
size, draw=1)`; Julia `Common.jl::load_split(key; draw=1)`, `cell_data(key, size; draw=1)`,
`split_path`. `Run_ICL.jl`: env `ICL_DRAW` (default 1); draw >= 2 writes
`results/icl/<ds>__<tag>__N<size>__<effort>__d<draw>.jsonl` (draw 1 keeps its names), every new record
carries `"draw"` (older records have none = draw 1), resume by (mol_id, repeat) per file as before;
the plan printout shows the draw and the first five training molecules of each dataset;
`ICL_PROBE_PLAN=1` without `ICL_PROBE` now prints the plan with the pending calls per model and exits
(before 2026-10-13 that combination fell through to the PAID main loop). `src/analyze_icl.py` parses
the draw from the file name (no suffix = 1) and writes a `draw` column into `cells.csv` (after
`reasoning`) and the per-molecule CSVs; a cell is (dataset, model, size, effort, draw).
`src/report_icl.py` (default and `--ladder`) and `src/analyze_calibration.py` use draw 1 only;
`src/report_icl.py --draws` writes `docs/RESULTS_DRAWS.md` + `results/icl/draws_cells.csv` (per
dataset x model x N: RMSE, Spearman, rho(UC1,|err|), PICP68 and RMSE50 = RMSE of the 50 % most
certain molecules by UC1 per draw -- since 2026-10-14 the exact square root of the expected MSE over random tie-breaking,
`src/selective_prediction.py`, section 7 -- mean and SD over the draws; LLM cells with coverage < 0.95 are
left out and listed). Baselines: `run_baselines.py --draw d` -> `<method>__<ds>__N<size>__d<d>.csv`
(+ meta with `draw`), `analyze_baselines.py` adds `draw` to `summary.csv` (sorted by draw first;
the 240 draw-1 rows are unchanged in content and order), `selfcheck.py` checks each cell against
its own draw. Verified 2026-10-13 without API calls: Julia dry run and plan for `ICL_DRAW=2`
(84 cells = 7 panel models x 4 datasets x 3 sizes, 37,800 calls at R = 3, all pending for draws 2
and 3), prompt parity Julia == Python for draws 1, 2 and 3 (the draw-2/3 training blocks differ
from draw 1, system and task messages identical), `analyze_icl.py` reproduces the previous
`cells.csv` / per-molecule CSVs byte for byte apart from the new column (same JSONL snapshot, old
vs new script), and `report_icl.py` default / `--ladder` outputs are identical to the old script's.

**Projected cost** (measured draw-1 cost of the low-effort cells at N = 30 / 60 / 120, R = 3, i.e.
450 calls per cell, 12 cells per model and draw; GPT-6 Astra has only lipophilicity N = 30 / 60 so
far: its other cells are GPT-6 Luna's measured cells x 60.3, the Astra / Luna cost ratio on those
two cells; 5,400 calls x its measured $0.0100 per call gives $54):

| model | per draw | draws 2 + 3 |
|---|---|---|
| Sonnet 5.5 | $27.61 | $55.21 |
| GPT-6 Astra | ~$51 (51-54) | ~$102 (102-108) |
| GPT 6.1 Sol | $7.29 | $14.57 |
| Gemini 3.8 Flash | $12.46 | $24.91 |
| Haiku 5.5 | $1.52 | $3.03 |
| GLM 5.3 Flash | $1.10 | $2.20 |
| GPT-6 Luna | $0.85 | $1.69 |
| **total** | **~$102 (102-105)** | **~$204 (204-210)** |

Launch plan: three detached processes by model group, each running draw 2 then draw 3 with
`ICL_THREADS=12 ICL_REPEATS=3 ICL_SIZES=30,60,120` and a per-run `ICL_BUDGET` (per Julia
invocation, i.e. per draw): sonnet55 (~$28 per draw, budget 40) | gpt6astra + gpt61sol (~$58-61
per draw, budget 85) | gem38flash + haiku55 + gpt6luna + glm53flash (~$16 per draw, budget 25).
Launched 2026-10-13 (see status note) (needs the project lead's go).

**Baselines on the draws** (no API cost; `run_baselines.py --methods tabpfn,rf,rf_desc,trivial
--datasets all --sizes 30,60,120 --draw 2|3` in `uc44`, then `analyze_baselines.py`): RMSE per draw
1 / 2 / 3 (mean, SD):

| dataset | method | N = 30 | N = 60 | N = 120 |
|---|---|---|---|---|
| lipophilicity | TabPFN | 1.17 / 1.05 / 1.10 (1.10, 0.06) | 1.01 / 1.01 / 1.01 (1.01, 0.00) | 0.94 / 0.94 / 0.87 (0.92, 0.04) |
| lipophilicity | rf_desc | 1.22 / 1.13 / 1.19 (1.18, 0.04) | 1.08 / 1.11 / 1.02 (1.07, 0.04) | 0.94 / 0.98 / 0.96 (0.96, 0.02) |
| BACE | TabPFN | 1.15 / 1.17 / 1.12 (1.14, 0.03) | 1.01 / 1.18 / 1.07 (1.08, 0.09) | 1.02 / 1.01 / 0.92 (0.99, 0.06) |
| BACE | rf_desc | 1.16 / 1.30 / 1.21 (1.22, 0.07) | 1.13 / 1.24 / 1.07 (1.15, 0.09) | 1.12 / 1.09 / 0.98 (1.06, 0.07) |
| antiviral | TabPFN | 0.85 / 0.82 / 0.83 (0.83, 0.01) | 0.85 / 0.78 / 0.77 (0.80, 0.05) | 0.79 / 0.70 / 0.72 (0.74, 0.04) |
| antiviral | rf_desc | 0.84 / 0.81 / 0.85 (0.83, 0.02) | 0.86 / 0.79 / 0.82 (0.82, 0.03) | 0.79 / 0.73 / 0.76 (0.76, 0.03) |
| ESOL | TabPFN | 0.94 / 1.02 / 1.09 (1.02, 0.07) | 0.90 / 1.00 / 1.03 (0.98, 0.07) | 0.78 / 0.97 / 0.71 (0.82, 0.13) |
| ESOL | rf_desc | 0.91 / 1.19 / 0.99 (1.03, 0.14) | 0.94 / 1.03 / 0.89 (0.95, 0.07) | 0.88 / 0.94 / 0.83 (0.88, 0.06) |

Reading: the context draw moves the baselines' RMSE by SD 0.00-0.14 at these sizes (largest on
ESOL), the same order as the draw-1 steps of the LLM learning curves between N = 30, 60 and 120
(|delta RMSE| 0.01-0.17), so single-draw small-N differences of < ~0.1 should not be read as
learning-curve shape; whether the LLMs are as draw-sensitive as the baselines is what draws 2 and 3
will show. TabPFN's draw-1 antiviral N = 60 point (0.855,
above its N = 30 value 0.845) is a draw effect too: draws 2 and 3 give 0.779 / 0.769, below their N = 30
values (0.819 / 0.827). Full tables
(also Spearman, rho, PICP68, RMSE50): `docs/RESULTS_DRAWS.md`.

**Status 2026-10-13 12:30:** draw-2 then draw-3 runs LAUNCHED as cmd.exe chains per model group (sonnet | gpt61sol | gem38flash+haiku55+gpt6luna+glm53flash; ICL_THREADS=12, R=3, budgets 40/15/25 per draw; logs `results/icl/run_draws_<group>_2026-10-13.log`); GPT-6 Astra's draws start automatically after its main run (`results/icl/wait_and_launch_draws_astra.ps1`, budget 60 per draw). Independent review (Opus 5.5): splits, runner, analysis and baselines verified correct; the runs demonstrably use the draw-2 contexts (prompt tokens +181 vs draw 1 for the same molecule).

**Status 2026-10-13 19:30: draws 2 and 3 COMPLETE for all 7 panel models** (168 draw cells, every cell
>= 445 of 450 unique (mol_id, repeat) pairs after the gap-fill; the 12 cells below 450 are the known
stochastic OpenAI endpoint rejections on antiviral plus a few Luna flex-tier failures). Measured cost of
draws 2 + 3: $180 (Astra $78, Sonnet $53, Gemini $34 incl. ~$2 of duplicated records where two Gemini
processes overlapped on antiviral N = 30/60 draw 3 -- `analyze_icl.py` deduplicates by (mol_id, repeat)),
the rest < $15). Operational notes: the four-model chain was relaunched as one chain per model at
16:00 (the combined chain was serial over 48 cells); Gemini, Luna and GLM draw 3 additionally got a
helper process working the dataset list in reverse order (`run_draws3b_*`), Luna a third one; GPT-6
Luna on the flex tier queued at 150-1,600 s per 450-call cell. Analysis regenerated: `cells.csv`,
per-molecule CSVs, `docs/RESULTS_ICL_R3.md`, `RESULTS_LADDER_R1.md` (now with `trunc` / `parse_rate`
columns and the Gemini `high` censoring note), `RESULTS_DRAWS.md` (3 draws for every LLM and baseline),
`RESULTS_CALIBRATION_R3.md` (draw 1, now including GPT-6 Astra), `selective_prediction.csv`,
figures `figs/fig1..fig5b` (`src/make_figs.py`, reviewed 2026-10-13).

Reading of the three draws (`results/icl/draws_cells.csv`): the draw-1 antiviral inversion (every LLM
worse at N = 60 than at N = 30) is a draw-1 artefact -- 7 of 7 models in draw 1, 0 of 7 in draws 2 and
3. Over the three draws the LLMs are LESS draw-sensitive than the baselines: median SD of RMSE over
draws per (dataset, N) is 0.01-0.06 for the LLMs against 0.02-0.09 for the baselines (largest for the
GP / MLP at N = 30), consistent with a prior-dominated predictor whose small-N output depends less on
which 30 molecules it is shown.
