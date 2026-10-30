#
# LLMs.jl -- replaces the in-house LLMs.jl of the utility library in this repository.
#
# The in-house file also defined private endpoints, further API keys and the model handles of other
# projects; the runners use none of them (they build their OpenRouter access objects from
# src/registry/models_v2.json, Common.jl::llm_access) and only need `key_openrouter`.
#
# Key: the environment variable OPENROUTER_API_KEY, or the line OPENROUTER_API_KEY=<key> of the
# git-ignored .env file in the repository root (an existing environment variable wins). The value is
# never printed or written anywhere; if it is missing, require_env stops with an error naming the
# variable.
#
include(joinpath(@__DIR__, "EnvConfig.jl"))
load_dotenv(normpath(joinpath(@__DIR__, "..", "..", "..", ".env")); quiet=true)
include(joinpath(@__DIR__, "LLMUtils.jl"))

key_openrouter = require_env("OPENROUTER_API_KEY")
