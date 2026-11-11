"""
Prompt construction and output parsing for the v2 in-context (ICL) experiment.

Blinding level 1 = real property name and units, real target values, real SMILES. Wording is
the v1 io prompt (archive/v1_2026-09_contaminated/src/Lipophilicity_Prompts.jl,
`input_output_prompting`), restructured into three messages (decision 2026-09-08):

  1. system   role + problem description (v1 text, dataset words substituted)
  2. user     the training block: header + one record per molecule ("SMILES: ...\\n<label>: ...")
              -- identical for every test molecule of a (dataset, size) cell and marked
              `cache_control: ephemeral`, so it is served from the prompt cache
  3. user     the prediction task for one test molecule + the JSON output contract (v1 text)

The in-context set of size N is the first N molecules of the split's training order, so the
sets are nested across the ladder and the N=1000 block is a prefix-extension of the N=500 one.

Parsing (= v1 analyze_uc.extract_json_block / coerce_float): take the LAST fenced ```json
block (fallback: greedy {...}), try raw, normalised (unicode minus, comments, trailing commas,
NaN/Infinity), then single->double quotes; prediction and uncertainty coerced to finite floats.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import DATASETS, load_dataset, load_split  # noqa: E402

_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FALLBACK_OBJ_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def system_prompt(key):
    d = DATASETS[key]
    return (
        "**Your Role**\n"
        f"You are a professional chemist with expert knowledge in {d['expertise']}.\n"
        f"You are tasked with predicting the {d['task']} of organic molecules based on "
        "their SMILES strings.\n\n"
        "**Problem Description**\n"
        "You will be provided with:\n"
        f"1. A training dataset of molecules with their SMILES and known {d['label']} values.\n"
        f"2. A test molecule (SMILES) for which you must predict the {d['label']}.\n\n"
        "Use your chemical knowledge and the relationships between molecular structure and the "
        f"{d['label']} in the training molecules to make an accurate prediction.\n")


def fmt_value(v):
    """Fixed 2 decimals for every dataset (== Julia fmt_value; decision 2026-10-09: removes the
    7-decimal bace.csv signature and the 10-decimal antiviral averaging artefacts)."""
    return f"{float(v):.2f}"


def training_block(key, train_rows):
    """train_rows: iterable of (smiles, value). One record per molecule, v1 extended-block style."""
    d = DATASETS[key]
    lines = ["**Training Data:**",
             f"{len(train_rows)} molecules from the same source as the test molecule, "
             f"with their SMILES and known {d['label']} values:", ""]
    for smi, val in train_rows:
        lines.append(f"SMILES: {smi}")
        lines.append(f"{d['label']}: {fmt_value(val)}")
        lines.append("")
    return "\n".join(lines)


def task_prompt(key, smiles):
    d = DATASETS[key]
    return (
        "**Prediction Task:**\n"
        f"Predict the {d['predict']} for the following molecule:\n"
        f"- SMILES: {smiles}\n\n"
        f"Also report your 1-sigma uncertainty for this prediction, in the same units as the "
        f"{d['label']}: the standard deviation of your predictive distribution, so that the "
        "measured value should fall within prediction +/- uncertainty about 68 % of the time.\n\n"
        "**Output (JSON)**\n"
        "Output the result as a single fenced JSON code block with EXACTLY this structure:\n"
        "```json\n{\n  \"prediction\": ...,\n  \"uncertainty\": ...\n}\n```\n"
        "Hard requirements:\n"
        f"- `prediction` is your predicted {d['label']}.\n"
        "- `uncertainty` is that 1-sigma value (a non-negative number).\n"
        "- The JSON block MUST be the LAST thing in your response (no text after the closing ```).\n")


def build_messages(key, train_rows, smiles):
    return [
        {"role": "system", "content": system_prompt(key)},
        {"role": "user", "content": [{"type": "text", "text": training_block(key, train_rows),
                                      "cache_control": {"type": "ephemeral"}}]},
        {"role": "user", "content": task_prompt(key, smiles)},
    ]


def cell_data(key, size, draw=1):
    """(test DataFrame, [(smiles, value)] of the first `size` training molecules of context draw
    `draw` (1 = data/splits/<key>_v2.json, the main split; 2, 3 = <key>_v2_draw<d>.json))."""
    df = load_dataset(key)
    df["mol_id"] = df.mol_id.astype(str)
    sp = load_split(key, draw)
    if size > sp["n_train_pool"]:
        raise ValueError(f"{key}: size {size} exceeds the training pool ({sp['n_train_pool']})")
    by_id = df.set_index("mol_id")
    train = [(by_id.loc[m, "smiles"], by_id.loc[m, "value"]) for m in sp["train_order_ids"][:size]]
    test = by_id.loc[sp["test_ids"]].reset_index()
    return test, train


def resolve_sizes(key, spec):
    """'min' | 'max' | 'all' | comma list of integers -> list of sizes valid for the dataset."""
    ladder = load_split(key)["sizes"]
    out = []
    for s in str(spec).split(","):
        s = s.strip()
        if s == "min":
            out.append(ladder[0])
        elif s == "max":
            out.append(ladder[-1])
        elif s == "all":
            out.extend(ladder)
        else:
            n = int(s)
            if n in ladder:
                out.append(n)
            elif n > ladder[-1]:
                pass                     # rung does not exist for this dataset (small pool)
            else:
                out.append(n)
    return sorted(dict.fromkeys(out))


def _normalise(s):
    s = s.replace("−", "-")
    s = re.sub(r"//[^\n]*", "", s)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bNaN\b", "null", s)
    s = re.sub(r"\bInfinity\b", "null", s)
    return s


def extract_json_block(text):
    if not text:
        return None
    candidates = _JSON_FENCE_RE.findall(text)[::-1]          # prefer the LAST fenced block
    if not candidates:
        m = _FALLBACK_OBJ_RE.search(text)
        if m:
            candidates = [m.group(0)]
    for cand in candidates:
        for variant in (cand, _normalise(cand), _normalise(cand).replace("'", '"')):
            try:
                obj = json.loads(variant)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def coerce_float(x):
    if x is None:
        return None
    if isinstance(x, str):
        x = x.replace("−", "-").strip()
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def parse_response(text):
    obj = extract_json_block(text)
    pred = coerce_float(obj.get("prediction")) if obj else None
    unc = coerce_float(obj.get("uncertainty")) if obj else None
    if unc is not None and unc < 0:
        unc = abs(unc)
    return dict(prediction=pred, uncertainty=unc, parse_ok=pred is not None and unc is not None)


def approx_tokens(text):
    """Rough token count (SMILES tokenise at ~1.2 chars/token, prose at ~4): use 2.5 overall."""
    return int(len(text) / 2.5)
