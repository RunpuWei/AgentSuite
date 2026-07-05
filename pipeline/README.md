# COBA: Component-Based Benchmark Auditing Pipeline

COBA is a taxonomy-driven auditing pipeline for identifying and analyzing issues in agent evaluation benchmarks.

It operationalizes a structured taxonomy of **benchmark task-level issues** and applies a two-stage process combining rule-based detection and LLM-based assessment, with human verification ([paper link (placeholder)](https://arxiv.org/abs/0000.00000)).


## Overview

COBA processes agent benchmark data to identify issues in benchmark tasks, tools, evaluation, and environments.

The pipeline consists of two stages:

- **Stage 1: Benchmark-aware rule-based filtering**  
  Applies deterministic rules to detect structural or misalignment issues

- **Stage 2: LLM-as-Judge assessment**  
  Uses an LLM to evaluate more subtle or semantic issues that are difficult to capture via rules

The pipeline is grounded in a structured taxonomy of agent benchmark issues, enabling systematic detection and categorization.

Human verification is applied to validate identified issues.

---
## Key Outputs

For each benchmark, COBA produces:

- Identified issue cases (per task)
- Filtering decisions (rule-based + LLM-based)
- Alignment with human-labeled ground truth
- Retention statistics across filtering stages

---

## Project Structure

```
coba_pipeline/
├── main.py                          # Main pipeline entry point
├── src/
│   ├── benchmark_filters/           # Rule-based filters
│   ├── bench_loaders/               # Data loaders for different benchmarks
│   ├── prompts/                     # LLM-as-Judge prompts
│   ├── metrics/                     # Metrics computation modules
│   ├── utils/                       # Utility functions and types
│   ├── data_loader.py
│   ├── download_model_trajectory.py   # Optional: fetch evaluation JSONL from Hugging Face
│   ├── llm_judge_filtering.py
│   └── rule_filtering_orchestrator.py
├── benchmark/                       # Benchmark evaluation results (user-provided)
├── human_labelled_ground_truth/     # Human-labeled ground truth for validation
├── pipeline_results/                # Output directory
├── requirements.txt                 # Python dependencies
├── setup_env.sh                    # Environment setup script
└── .env.example                    # API configuration template
```

## Prerequisites

- Python 3.11+
- Conda or Miniconda
- **Benchmark evaluation results** (see [Benchmark Data Setup](#benchmark-data-setup) below)
- OpenAI-compatible API endpoint for LLM-as-Judge filtering (optional if using precomputed results)

## Installation

### Quick Setup (Recommended)

Run the automated setup script:

```bash
bash setup_env.sh
```

This will:
1. Create a conda environment named `coba_pipeline` with Python 3.11
2. Install all required dependencies
3. Verify the installation

### Manual Setup

```bash
# Create and activate conda environment
conda create -n coba_pipeline python=3.11 -y
conda activate coba_pipeline

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### API Configuration

The pipeline requires API credentials for LLM-as-Judge filtering:

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and configure your API credentials:
   ```bash
   API_KEY=your_api_key_here
   BASE_URL=https://api.openai.com/v1
   ```

   **Note**: Currently only OpenAI-compatible API endpoints are supported (e.g., OpenAI, Azure OpenAI, vLLM, LiteLLM).

   **Optional**: If you have precomputed LLM-as-Judge results in the `cached_results/` directory, you can skip API configuration and use `--precomputed-results` flag (see [Usage Examples](#examples) below).

### Benchmark Data Setup

**IMPORTANT**: This repository contains only the filtering pipeline. You must provide evaluation JSONL under `benchmark/`, or fetch public trajectories with `python src/download_model_trajectory.py` (see that script’s docstring for Hugging Face repo ids).

#### Directory Structure

Place your benchmark evaluation results in the `benchmark/` directory with the following structure:

```
benchmark/
├── <benchmark_name>-evaluation/
│   ├── <model_1>.jsonl
│   ├── <model_2>.jsonl
│   ├── <model_3>.jsonl
│   └── ...
```

Example:
```
benchmark/
├── tau-bench-evaluation/
│   ├── gpt-4o.jsonl
│   ├── claude-4-sonnet.jsonl
│   └── gemini-2.0-pro.jsonl
├── DrafterBench-evaluation/
│   ├── gpt-4o.jsonl
│   └── ...
```

#### JSONL Format

Each line in the JSONL file should be a JSON object with the following required fields:

```json
{
  "model_path": "anthropic/claude-4.5-sonnet",
  "benchmark_name": "tau-bench",
  "task_name": "airline",
  "messages": [
    {"role": "system", "content": "...", "turn_idx": 0},
    {"role": "user", "content": "...", "turn_idx": 1},
    {"role": "assistant", "content": "...", "turn_idx": 2, "tool_calls": [...]},
    {"role": "tool", "content": "...", "turn_idx": 3, "tool_call_id": "...", "name": "..."}
  ],
  "meta": {
    "id": "airline_1",
    "is_correct": true
  },
  "eval_result": {
    "score": 1.0,
    "passed": true
  }
}
```

**Required fields:**
- `model_path` or `model_name`: Model identifier
- `benchmark_name`: Name of the benchmark
- `task_name`: Task/subtask identifier (optional for some benchmarks)
- `messages`: Full conversation trace including tool calls
- `meta`: Metadata object with at least an `id` field
- `eval_result`: Evaluation results with success indicators (`score`, `passed`, `is_correct`, etc.)

**Note**: This pipeline does NOT run benchmark evaluations. You need to:
1. Run your models on the target benchmarks separately
2. Format the results according to the above specification
3. Place the JSONL files in the appropriate `benchmark/<benchmark_name>-evaluation/` directory

## Quick Start Workflow

### Typical Usage Flow

1. **Prepare benchmark data**: Place evaluation results in `benchmark/<benchmark>-evaluation/`
2. **Option A - With API access**: Configure `.env` with API credentials
3. **Option B - Without API access**: Use precomputed results from `cached_results/`
4. **Run the pipeline**: Execute `python main.py` with appropriate flags
5. **Review results**: Check output files in `pipeline_results/`

### Running Without API Access

If you don't have API credentials but want to test the pipeline, use precomputed results:

```bash
# Example: Run pipeline with cached LLM-as-Judge results
python main.py \
  --target-benchmark tau-bench \
  --precomputed-results cached_results/tau-bench_20250924_011852_filtering_summary.csv
```

This will:
- ✅ Skip Step 2 (LLM-as-Judge filtering)
- ✅ Use cached filtering decisions
- ✅ Still generate filtering summaries and retention/alignment logs
- ❌ Cannot modify LLM filtering decisions

## Usage

### Basic Usage

```bash
conda activate coba_pipeline
python main.py --target-benchmark <benchmark_name>
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--target-benchmark` | str[] | all | Target benchmark(s) to process |
| `--llm-model` | str | google/gemini-2.5-pro-thinking-on | LLM model for Step 2 filtering |
| `--num-proc` | int | 1 | Number of parallel processes for LLM filtering |
| `--llm-max-samples` | int | None | Maximum samples to process (for testing) |
| `--rebuttal` | flag | false | Enable rebuttal in LLM-as-Judge filtering |
| `--precomputed-results` | str | None | Path to CSV with precomputed filtering results |

### Examples

**Process a single benchmark:**
```bash
python main.py --target-benchmark tau-bench
```

**Process multiple benchmarks with parallel LLM filtering:**
```bash
python main.py --target-benchmark tau-bench DrafterBench --num-proc 8
```

**Use a specific LLM model:**
```bash
python main.py \
  --target-benchmark tau-bench \
  --llm-model openai/gpt-4o
```

**Test with limited samples:**
```bash
python main.py \
  --target-benchmark tau-bench \
  --llm-max-samples 100
```

**Use precomputed LLM-as-Judge results (skip API calls):**
```bash
python main.py \
  --target-benchmark tau-bench \
  --precomputed-results cached_results/tau-bench_20250924_011852_filtering_summary.csv
```

**Note**: The `cached_results/` directory contains precomputed LLM-as-Judge filtering results. Using this option allows you to run the pipeline without API credentials, as it will skip Step 2 (LLM-as-Judge) and use the cached results instead. This is useful for:
- Testing the pipeline without API access
- Reproducing previous filtering runs
- Avoiding redundant API calls

## Output

The pipeline generates several output files in the `pipeline_results/` directory:

1. **Filtering Summary CSV**: `<benchmark>_<timestamp>_filtering_summary.csv`
   - Detailed filtering results for each question
   - Shows which filters passed/failed for each sample

2. **Log File**: `<benchmark>_<timestamp>_pipeline.log`
   - Detailed execution logs
   - Retention ratio and human-alignment summaries
   - Error messages and warnings

## Metrics

The pipeline computes the following metrics at each filtering stage:

- **Precision/Recall**: Alignment with human-labeled ground truth data
- **Retention Ratio**: Proportion of samples retained after filtering

## Supported Benchmarks

The pipeline currently supports:
- ACEBench
- BFCL v3
- BFCL v4 (expanded sample)
- ComplexFuncBench
- tau-bench
- tau2-bench
- DrafterBench
- NexusBench (expanded sample)

## Development

### Adding a New Benchmark

1. Create a loader in `src/bench_loaders/<benchmark>_loader.py`
2. Create a filter in `src/benchmark_filters/`
3. Create prompts in `src/prompts/<benchmark>_prompt.py`
4. Add the benchmark to `src/utils/types.py` in the `Benchmark` enum



## Citation

If you use the COBA pipeline, please cite:

```bibtex
@article{agentsuite2026,
  title   = {Toward More Reliable Agent Evaluation: A Component-Based Benchmark Auditing Pipeline},
  author  = {Hyewon Suh, Seojune Lee, Binfei Ji, Rishi Khare, Basit Khan, Hyunjun Kim, Tianyi Zhang, Venkat Krishna Srinivasan, Peter Belcak, Shizhe Diao, Pavlo Molchanov, Yingyan Celine Lin, Zhen Dong},
  journal = {arXiv preprint arXiv:0000.00000(placeholder)},
  year    = {2026}
}