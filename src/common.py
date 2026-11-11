"""
Shared paths, dataset registry and helpers for the v2 uncertainty study (project 44).

Datasets are the prepared screening files of the memorization paper (project 45), copied
verbatim to data/<key>.csv. Their row order is the SOURCE file order (no shuffling, no sorting
by us): row index == position in the MoleculeNet / DeepChem / Polaris file minus dropped rows.
Schema: mol_id, smiles, value, smiles_random, smiles_blind, value_blind (+ extras for freesolv).
"""
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
TEST_SETS = os.path.join(DATA, "test_sets")
RESULTS = os.path.join(ROOT, "results")
CONTAM = os.path.join(RESULTS, "contamination")
MEMO45 = os.path.join(ROOT, "src", "memo45")
P45 = os.path.abspath(os.path.join(ROOT, "..", "45_LLM_Memorization_Paper"))

# The four datasets of v2 (decision 2026-09-08: ESOL replaces FreeSolv as the contaminated
# counterpart because FreeSolv's 642 molecules cannot supply a separate 1000-sample training
# pool; FreeSolv is discarded from all experiments and from the corpus screen).
DATASET_KEYS = ["lipophilicity", "bace", "antiviral", "esol"]
ALL_KEYS = DATASET_KEYS
LEGACY_KEYS = ["freesolv"]   # data and early check results kept on disk for reference only

# L1 ("level 1", real property name, real values, real SMILES) prompt building blocks in the
# wording of the v1 io prompt (archive/v1_2026-09_contaminated/src/Lipophilicity_Prompts.jl):
#   expertise   -> "You are a professional chemist with expert knowledge in {expertise}."
#   task        -> "You are tasked with predicting the {task} of organic molecules ... based on their SMILES strings."
#   label       -> line label of the target in the training block and the JSON hard requirements
#   predict     -> "Predict the {predict} for the following molecule"
DATASETS = {
    "lipophilicity": dict(
        name="Lipophilicity (MoleculeNet, AZ/ChEMBL)", unit="logD (pH 7.4)",
        expertise="physical chemistry and lipophilicity prediction",
        task="lipophilicity (octanol/water distribution coefficient logD at pH 7.4)",
        medium="", label="logD (pH 7.4)",
        predict="octanol/water distribution coefficient logD at pH 7.4",
        contamination_status="clean for all 22 panel models (45 paper)"),
    "bace": dict(
        name="BACE (MoleculeNet)", unit="pIC50",
        expertise="medicinal chemistry and BACE-1 inhibitor potency prediction",
        task="BACE-1 inhibitory potency (pIC50)", medium="", label="BACE-1 pIC50",
        predict="BACE-1 inhibitory potency (pIC50)",
        contamination_status="clean for all 22 panel models (45 paper)"),
    "antiviral": dict(
        name="Antiviral potency (ASAP/Polaris 2025)", unit="pIC50",
        expertise="medicinal chemistry and SARS-CoV-2 main protease (Mpro) inhibitor potency prediction",
        task="SARS-CoV-2 main protease (Mpro) inhibitory potency (pIC50)", medium="",
        label="SARS-CoV-2 Mpro pIC50", predict="SARS-CoV-2 main protease (Mpro) inhibitory potency (pIC50)",
        contamination_status="clean for all 22 panel models (45 paper); recency control"),
    "freesolv": dict(
        name="FreeSolv / SAMPL", unit="kcal/mol",
        expertise="physical chemistry and solvation thermodynamics",
        task="hydration free energy (kcal/mol)", medium="in water",
        label="Hydration free energy (kcal/mol)", predict="hydration free energy (kcal/mol)",
        contamination_status="CONTAMINATED counterpart: 14/22 models flagged (45 paper)"),
    "esol": dict(   # the ESOL file also carries an ESOL-*predicted* value; the label names the MEASURED one
        name="ESOL / Delaney", unit="log(mol/L)",
        expertise="physical chemistry and solubility prediction",
        task="measured aqueous solubility logS (log10 of the solubility in mol/L)", medium="",
        label="measured logS (log10 mol/L)",
        predict="measured aqueous solubility logS (log10 of the solubility in mol/L)",
        contamination_status="CONTAMINATED counterpart: 15/22 models flagged (45 paper)"),
}

REGISTRY = os.path.join(ROOT, "src", "registry")
MODELS_V2 = os.path.join(REGISTRY, "models_v2.json")


def load_registry():
    """v2 model registry (src/registry/models_v2.json): {tag: entry}, in file order (= panel order)."""
    return {m["tag"]: m for m in json.load(open(MODELS_V2, encoding="utf8"))["models"]}


# Model panel for v2 (decision 2026-10-09 after the probes): the in_panel entries of
# src/registry/models_v2.json, in registry order (haiku55, sonnet55, gpt61sol, gem38flash, gpt6luna,
# glm53flash). Dropped: opus55 (refuses the ICL prompt), dsv41flash (account privacy setting), kimik3 (cost).
PANEL = [t for t, m in load_registry().items() if m.get("in_panel")]


def load_registry45():
    models = {m["tag"]: m for m in json.load(open(os.path.join(MEMO45, "models_45.json")))["models"]}
    dsets = {d["key"]: d for d in json.load(open(os.path.join(MEMO45, "datasets_45.json")))["datasets"]}
    return models, dsets


def load_dataset(key):
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA, f"{key}.csv"))
    df["row_index"] = range(len(df))          # 0-based position in the source file
    return df


def load_test_set(key, rule="v2test150"):
    import pandas as pd
    return pd.read_csv(os.path.join(TEST_SETS, f"{key}_{rule}.csv"))


def split_path(key, draw=1):
    """data/splits/<key>_v2.json for draw 1 (the main split), <key>_v2_draw<d>.json for the
    additional context draws d = 2, 3 (src/make_splits.py --draws 3; same test set and pool)."""
    draw = int(draw)
    if draw < 1:
        raise ValueError(f"draw must be >= 1, got {draw}")
    name = f"{key}_v2.json" if draw == 1 else f"{key}_v2_draw{draw}.json"
    return os.path.join(DATA, "splits", name)


def load_split(key, draw=1):
    """v2 split: {test_ids, train_order_ids, sizes, ...} from data/splits/<key>_v2.json (draw 1)
    or data/splits/<key>_v2_draw<draw>.json (additional context draws; identical test_ids, sizes and
    training pool, independent permutation of the pool)."""
    sp = json.load(open(split_path(key, draw), encoding="utf8"))
    if int(sp.get("draw", 1)) != int(draw):
        raise ValueError(f"{split_path(key, draw)}: stored draw {sp.get('draw')} != requested {draw}")
    return sp


DOTENV = os.path.join(ROOT, ".env")      # optional, git-ignored: NAME=value lines


def dotenv_value(name):
    """Value of NAME in the git-ignored .env file in the repository root (NAME=value lines, optional
    `export ` prefix and quotes), or None. Never prints the value."""
    if not os.path.exists(DOTENV):
        return None
    with open(DOTENV, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip().removeprefix("export ").strip() == name:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                return value or None
    return None


def openrouter_key():
    """Read the OpenRouter key: the environment variable OPENROUTER_API_KEY first, else the line
    OPENROUTER_API_KEY=... of the git-ignored .env file in the repository root. Never print the value
    or commit the .env file."""
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if k:
        return k
    k = dotenv_value("OPENROUTER_API_KEY")
    if k:
        return k
    raise RuntimeError(f"no OpenRouter key: set the environment variable OPENROUTER_API_KEY or put the line "
                       f"OPENROUTER_API_KEY=<key> into {DOTENV} (git-ignored)")


NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_number(txt):
    """Last number in the reply, scientific notation first (== 45/run_reasoning_ladder.parse_number)."""
    t = txt.replace("\u2212", "-").replace("\u2013", "-").replace(",", "")
    m = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+", t)
    if m:
        try:
            return float(m[-1])
        except ValueError:
            pass
    m = NUM.findall(t)
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def canonical_forms(smiles):
    """Alternative string forms of the same molecule for corpus search.

    Returns dict form -> string. Forms that coincide with an earlier one are dropped so that
    each distinct string is queried once.
    """
    forms = {"verbatim": smiles.strip()}
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            forms["canonical_iso"] = Chem.MolToSmiles(mol, isomericSmiles=True)
            forms["canonical_noiso"] = Chem.MolToSmiles(mol, isomericSmiles=False)
            forms["inchikey"] = Chem.MolToInchiKey(mol)
    except Exception:
        pass
    out, seen = {}, set()
    for k, v in forms.items():
        if v and v not in seen:
            out[k] = v
            seen.add(v)
    return out
