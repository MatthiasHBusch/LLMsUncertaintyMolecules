#
# Zero-shot recall prompt of the memorization paper (45/src/Run_ZeroShot.jl / run_budget_sweep.py,
# with the "3 significant figures" sentence), shared by Run_ZeroShot_TestSet.jl and Probe_Models.jl.
# Dataset words come from the 45 dataset registry (Common.jl::load_datasets45). Include after Common.jl.
#

"System and user message of the 45 controlled map (dataset words from the 45 registry)."
function zs_prompts(ds::AbstractDict, smiles::AbstractString)
    expert = ds["prompt_expert"]; dsname = ds["prompt_dsname"]
    prop = ds["prompt_property"]; unit = ds["prompt_unit"]
    sys = "You are an $(expert) and know the $(dsname) dataset very well. You are given a SMILES " *
          "string of a molecule. Your task is to predict the $(prop) of that molecule in $(unit) as " *
          "reported in that dataset. Provide only the numerical value as output, without any " *
          "additional text. Report the value with 3 significant figures."
    usr = "What is the $(prop) in $(unit) of the molecule with the following SMILES string in the " *
          "$(dsname) dataset: $(smiles)? Provide only the numerical value as output, without any " *
          "additional text. Report the value with 3 significant figures."
    return sys, usr
end

function zs_conversation(ds::AbstractDict, smiles::AbstractString)
    s, u = zs_prompts(ds, smiles)
    return Tuple[("system", s), ("user", u)]
end

"Test set data/test_sets/<key>_<rule>.csv (mol_id as String)."
function load_test_set(key::AbstractString, rule::AbstractString)
    df = DataFrame(CSV.File(joinpath(DATA_DIR, "test_sets", "$(key)_$(rule).csv")))
    df.mol_id = string.(df.mol_id)
    return df
end
