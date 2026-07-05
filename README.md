# AgentSuite: Toward More Reliable Agent Evaluation via a Component-Based Benchmark Auditing Pipeline

AgentSuite is a unified evaluation framework for LLM agents on tool-use and interactive benchmarks.

It provides **one-touch execution** of diverse benchmarks, along with **cleaned and filtered benchmark variants** that mitigate identifiable evaluation issues while preserving comparability with original benchmark settings.

AgentSuite provides:
- One-touch execution of multiple LLM agent benchmarks
- Cleaned and filtered benchmark variants for more reliable evaluation
- Support for OpenAI-compatible API endpoints

Issues are identified using our **COBA (Component-Based Benchmark Auditing) pipeline**, grounded in a structured taxonomy of agent benchmark issues ([paper](https://openreview.net/forum?id=2Exmr1eIKZ)).

📊 **Live leaderboard:** [AgentSuite Leaderboard on Hugging Face Spaces](https://huggingface.co/spaces/AgentSuite/AgentSuite) — Original vs. Filtering scores across all benchmarks. To add your model, submit a trajectory ([submission guide](./leaderboard/README.md)).

> ⚠️ Cleaned variants improve evaluation reliability by addressing identifiable and verifiable issues (e.g., malformed tools, incorrect ground truth), but do not attempt to exhaustively correct all possible sources of noise.

---

## COBA Pipeline

AgentSuite includes the **COBA (Component-Based Benchmark Auditing) pipeline** (located in `pipeline/`), used to identify and analyze issues in benchmark tasks.

The pipeline operationalizes a structured taxonomy of agent benchmark issues and is used to produce the cleaned and filtered benchmark variants, with human verification and targeted fixes applied where appropriate.

For more details, see the [`pipeline/` README](./pipeline/README.md).



## Setup

```bash
# 1. Create conda environment and install all dependencies
bash setup.sh

# 2. Activate the environment
conda activate agentsuite

# 3. Configure API credentials
cp .env.example .env
# Edit .env with your API_KEY and BASE_URL
```

**Required `.env` fields:**
```bash
API_KEY=your_api_key_here
BASE_URL=your_openai_compatible_endpoint_url

# Optional: separate endpoint for user simulation / judge models
USER_API_KEY=your_user_model_key_here
USER_BASE_URL=your_user_model_endpoint_url

# Optional: required for CFBench
RAPID_API_KEY=your_rapid_api_key_here
```

**Verify available models:**
```bash
python list_available_models.py
```

## Usage

```bash
# List available benchmarks
python run_benchmarks.py --list-benchmarks

# Run a single benchmark
python run_benchmarks.py "deepseek-ai/DeepSeek-V3" --benchmark cfbench

# Run all benchmarks
python run_benchmarks.py "deepseek-ai/DeepSeek-V3"

# Quick validation with limited samples
python run_benchmarks.py "deepseek-ai/DeepSeek-V3" --benchmark acebench --limit 32

# Full example with all options
python run_benchmarks.py "deepseek-ai/DeepSeek-V3" \
    --benchmark taubench \
    --proc-num 8 \
    --temperature 0 \
    --user-model "openai/gpt-4o-20240806" \
    --output-dir results/my_run \
    --limit 20
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `model_name` | (required) | Model name (e.g., `openai/gpt-4o-20240806`) |
| `--benchmark` | `all` | Benchmark to run (see table below) |
| `--proc-num` | `4` | Number of parallel processes/threads |
| `--temperature` | `0.0` | Generation temperature |
| `--user-model` | `openai/gpt-4o-20240806` | Model for user simulation / judge |
| `--output-dir` | `./results` | Output directory |
| `--limit` | (none) | Limit samples per benchmark for quick validation |
| `--concurrent` | off | Run benchmarks in parallel |
| `--fixed` | off | Fixed split: use fixed task sets/data (see below; `--cleaned` is a deprecated alias) |

### Available Benchmarks

| Benchmark | Description |
|---|---|
| `acebench` | Agent capability evaluation (normal + special + agent) |
| `cfbench` | Complex multi-step function calling (requires `RAPID_API_KEY`) |
| `taubench` | Tool-agent-user interaction (retail + airline) |
| `tau2bench` | Conversational agents (retail + airline + telecom) |
| `bfcl_v3` | BFCL single-turn + multi-turn (prompting mode) |
| `bfcl_v3_fc` | BFCL single-turn + multi-turn (function-calling mode) |
| `bfcl_v4` | BFCL agentic tasks (prompting mode) |
| `bfcl_v4_fc` | BFCL agentic tasks (function-calling mode) |
| `drafterbench` | Technical drawing revision tasks |
| `nexusbench` | Function calling benchmark |
| `multichallenge` | Multi-domain conversation evaluation |

### BFCL Function-Calling (FC) Mode

BFCL supports two inference modes:

- **Prompting mode** (`bfcl_v3` / `bfcl_v4`): Functions are described as text in the system prompt. The model returns function calls as plain text that is parsed post-hoc.
- **Function-calling mode** (`bfcl_v3_fc` / `bfcl_v4_fc`): Functions are passed as structured `tools` via the OpenAI tool-calling API. The model returns structured `tool_calls` objects.

FC mode requires the API endpoint to support OpenAI-compatible tool calling. A lightweight probe request is sent before evaluation starts — if the endpoint rejects it, the run aborts early with a clear error message.

```bash
# Prompting mode (default)
python run_benchmarks.py "deepseek-ai/DeepSeek-V3" --benchmark bfcl_v3

# Function-calling mode
python run_benchmarks.py "deepseek-ai/DeepSeek-V3" --benchmark bfcl_v3_fc

# Both modes
python run_benchmarks.py "deepseek-ai/DeepSeek-V3" --benchmark bfcl_v3,bfcl_v3_fc
```

> **Note:** `--benchmark all` runs prompting mode only. FC variants must be specified explicitly.

### Evaluation variants

Flagged tasks for each benchmark live in a single manifest, `<bench>_issues.csv`
(the source of truth), and feed two cleaned variants:

- **Filtered** — drops the flagged tasks and re-scores the original results over
  what remains. No reruns, original data. This is the launch variant.
- **Fixed** (`--fixed`, deprecated alias `--cleaned`) — applies targeted repairs
  on fixed data and drops the unfixable tasks.

See [`eval_splits.md`](./eval_splits.md) for the manifest schema, the
`derive_split_ids.py` generator, and per-benchmark notes.

When `--fixed` is enabled, AgentSuite switches to fixed task sets/data:

| Benchmark | Fixed data | Filtered tasks |
|---|---|---|
| `acebench` | `data_all_fixed/` | 1023 → 999 |
| `cfbench` | (same data) | 1000 → 871 |
| `taubench` | `envs_fixed/` (tasks, tools, data) | airline 50→39, retail 115→109 |
| `tau2bench` | `domains_fixed/` | airline 50→31, retail 114→102, telecom 114→104 |
| `bfcl_v3`/`bfcl_v3_fc`/`bfcl_v4`/`bfcl_v4_fc` | `data_fixed/` | varies by category |
| `drafterbench` | not supported due to upstream fix† | |
| `nexusbench`/`multichallenge` | not yet supported | errors if used |

† DrafterBench issues identified in our analysis were reported and have since been fixed in the upstream repository; no additional filtering is applied here.

```bash
# Run the Fixed variant
python run_benchmarks.py "deepseek-ai/DeepSeek-V3" --benchmark acebench --fixed
```

### Note on `--user-model`

CFBench, ACEBench, and MultiChallenge officially use `gpt-4o-2024-08-06` as the evaluation / user simulation model. Using a different `--user-model` will trigger a warning and may produce results that don't align with official leaderboards.

## Results

Results are saved to the output directory (`--output-dir`), organized by benchmark name:

```
results/
├── ACEBench/
│   ├── generate_output.log
│   ├── eval_output.log
│   └── result_all/ ...
├── CFBench/
│   ├── output.log
│   ├── full-1000.jsonl
│   └── metrics.log
├── TauBench/
│   ├── output.log          # includes acc summary
│   └── *.json              # per-environment results
├── BFCL_v3/                # prompting mode
│   ├── generate_output.log
│   ├── eval_output.log
│   ├── result/ ...
│   └── score/ ...
├── BFCL_v3_FC/             # function-calling mode (if requested)
│   ├── generate_output.log
│   ├── eval_output.log
│   ├── result/ ...
│   └── score/ ...
├── summary.json
└── summary.txt
```

## Included Benchmarks

| Benchmark | Reference |
|---|---|
| ACEBench | [arXiv:2501.12851](https://arxiv.org/abs/2501.12851) |
| ComplexFuncBench | [arXiv:2501.10132](https://arxiv.org/abs/2501.10132) |
| TauBench | [arXiv:2406.12045](https://arxiv.org/abs/2406.12045) |
| Tau2Bench | [tau2-bench](./tau2-bench/README.md) |
| BFCL | [gorilla.cs.berkeley.edu](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) |
| DrafterBench | [arXiv:2507.11527](https://arxiv.org/abs/2507.11527) |
| NexusBench | [github.com/nexusflowai/NexusBench](https://github.com/nexusflowai/NexusBench) |
| MultiChallenge | [multi_challenge](./multi_challenge/README.md) |


## Citation

If you use AgentSuite, please cite:

```bibtex
@misc{agentsuite2026,
  title  = {AgentSuite: Toward More Reliable Agent Evaluation via a Component-Based Benchmark Auditing Pipeline},
  author = {Hyewon Suh, Binfei Ji, Seojune Lee, Rishi Khare, Basit Khan, Hyunjun Kim, Tianyi Zhang, Venkat Krishna Srinivasan, Peter Belcak, Shizhe Diao, Pavlo Molchanov, Yingyan Celine Lin, Zhen Dong},
  year   = {2026},
  url    = {https://openreview.net/forum?id=2Exmr1eIKZ}
}