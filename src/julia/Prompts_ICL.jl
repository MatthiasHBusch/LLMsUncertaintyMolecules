#
# Prompts of the v2 in-context experiment: io protocol, blinding level 1.
#
# Same conventions as the transfer-learning / v1 prompt files (Lipophilicity_Prompts.jl):
# a `prompts` dictionary of templates with <placeholders>, filled by `fill_prompt`. The
# wording is the v1 `input_output_prompting` prompt, revised 2026-10-09 (see docs/EXPERIMENT_DESIGN_v2.md
# section 4: 1-sigma defined, SARS-CoV-2 Mpro named, logS/logD wording, 2-decimal values); the structural change (decision
# 2026-09-08) is that the training data is its own user message BEFORE the task prompt,
# tagged "ephemeral" so the identical block of a (dataset, size) cell is served from the
# prompt cache, and that the training block lists one record per molecule (the v1
# extended-block format) instead of two parallel lists.
#
# Conversation = [("system", ...), ("user", training_block, "ephemeral"), ("user", task)]
#
prompts = Dict{String,Dict{String,String}}()
prompts["io_level1"] = Dict{String,String}()

prompts["io_level1"]["system"] = """
**Your Role**
You are a professional chemist with expert knowledge in <expertise>.
You are tasked with predicting the <task> of organic molecules based on their SMILES strings.

**Problem Description**
You will be provided with:
1. A training dataset of molecules with their SMILES and known <label> values.
2. A test molecule (SMILES) for which you must predict the <label>.

Use your chemical knowledge and the relationships between molecular structure and the <label> in the training molecules to make an accurate prediction.
"""

prompts["io_level1"]["training_header"] = """
**Training Data:**
<n_train> molecules from the same source as the test molecule, with their SMILES and known <label> values:
"""

prompts["io_level1"]["training_record"] = """
SMILES: <smiles>
<label>: <value>
"""

prompts["io_level1"]["prediction"] = """
**Prediction Task:**
Predict the <predict> for the following molecule:
- SMILES: <smiles>

Also report your 1-sigma uncertainty for this prediction, in the same units as the <label>: the standard deviation of your predictive distribution, so that the measured value should fall within prediction +/- uncertainty about 68 % of the time.

**Output (JSON)**
Output the result as a single fenced JSON code block with EXACTLY this structure:
```json
{
  "prediction": ...,
  "uncertainty": ...
}
```
Hard requirements:
- `prediction` is your predicted <label>.
- `uncertainty` is that 1-sigma value (a non-negative number).
- The JSON block MUST be the LAST thing in your response (no text after the closing ```).
"""

"Replace every <key> in `prompt` by data[key]; warn about placeholders left over (as in v1)."
function fill_prompt(prompt::String, data::Dict)
    for (k, v) in data
        prompt = replace(prompt, "<" * k * ">" => string(v))
    end
    if occursin(r"<[a-z_]+>", prompt)
        println("Warning: unfilled placeholders: ", [m.match for m in eachmatch(r"<[a-z_]+>", prompt)])
    end
    return prompt
end

"Dataset words for the templates (medium gets its leading space here)."
function dataset_words(key::AbstractString)
    d = DATASETS[key]
    return Dict{String,Any}("expertise" => d["expertise"], "task" => d["task"],
                "medium" => isempty(d["medium"]) ? "" : " " * d["medium"],
                "label" => d["label"], "predict" => d["predict"])
end

system_prompt(key) = fill_prompt(prompts["io_level1"]["system"], dataset_words(key))

"Value as written in the prompt: fixed 2 decimals for every dataset (decision 2026-10-09).
Removes float artefacts (-4.2860000000000005), the 7-decimal signature of MoleculeNet bace.csv and
the 10-decimal stereo-averaging artefacts of the antiviral file; rounding error <= 0.005."
fmt_value(v) = @sprintf("%.2f", Float64(v))

"Training block: header + one record per (smiles, value), blank line between records."
function training_block(key::AbstractString, train::Vector)
    w = dataset_words(key)
    w["n_train"] = length(train)
    parts = [fill_prompt(prompts["io_level1"]["training_header"], w)]
    for (smi, val) in train
        d = copy(w); d["smiles"] = smi; d["value"] = fmt_value(val)
        push!(parts, fill_prompt(prompts["io_level1"]["training_record"], d))
    end
    return join(parts, "\n")
end

function task_prompt(key::AbstractString, smiles::AbstractString)
    d = dataset_words(key); d["smiles"] = smiles
    return fill_prompt(prompts["io_level1"]["prediction"], d)
end

"The three-message conversation for one test molecule (LLMUtils tuple convention)."
function build_conversation(key::AbstractString, train::Vector, smiles::AbstractString)
    return Tuple[("system", system_prompt(key)),
                 ("user", training_block(key, train), "ephemeral"),
                 ("user", task_prompt(key, smiles))]
end
