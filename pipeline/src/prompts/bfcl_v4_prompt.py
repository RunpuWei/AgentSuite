import argparse
import json
import os
import re
from typing import Any, Dict, List

from src.utils.types import BFCLv4Question, LLMJudgeStep


DEFAULT_FILTERING_PROMPT = """"""

MEMORY_FILTERING_PROMPT = """
You are an expert evaluator for **Berkeley Function Calling Leaderboard (BFCL) V4 Agentic, Part 2: Memory**, which examines an LLM agent's ability to correctly recall, reference, and integrate information from prior conversation context (memory) to answer follow-up questions that depend on earlier dialogue.
Your task is to **determine whether the given sample is fundamentally flawed**, meaning that even a perfect agent with accurate and complete memory access could not reasonably derive the correct answer from the provided conversation context (e.g., the ground-truth contradicts the memory, the memory lacks necessary information, or the question is logically incoherent).

You will be provided with the following information:

* **Question**: The question that was asked to the LLM model. It often requires accessing previous conversation memory to answer.
* **Ground-Truth Answer**: The canonical answer string(s) that must be contained (after normalization) in a model's output for it to be marked correct. If multiple ground-truth answers are provided, matching any one of them is sufficient for the sample to be considered correct.
  Normalization converts text to lowercase and strips punctuation marks such as ,./-\\_*^() so that superficial differences (e.g., "eiffel-tower" vs "Eiffel Tower") do not affect scoring.
* **Reference Sources**: A full or partial excerpt of previous conversation where the required information is located.
* **Conversation Memory**: A full multi-turn dialogue history between the user and the model, representing the memory state that the model is expected to recall from.

A sample is **flawed** if it exhibits one or more of the issues described below.

## Flaw Categories

Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.

### Question (task specification)

This category covers flaws in how the follow-up question is posed relative to what can be answered from memory.

* **Incoherent or unanswerable question**: The question is logically garbled, self-contradictory, or cannot be answered under any reasonable reading of the dialogue.
* **Underspecified or multiply-answerable question**: The question admits several materially different correct answers supported by the memory (or by fair interpretation of it), yet the benchmark commits to a single ground-truth without adequate grounding to prefer one over the others.

### Conversation memory and reference context

This category covers flaws in the dialogue the agent is expected to recall from, or in how reference excerpts relate to that dialogue, which can make the item unfair even for a perfect memory-based agent.

* **Insufficient context in memory**: The conversation memory (and the reference excerpts, taken together) does not contain, or does not uniquely support, the facts needed to justify the ground-truth answer.
* **Contradiction between ground-truth and memory**: The ground-truth answer conflicts with explicit statements or clearly established facts in the conversation memory.
* **Reference material misaligned with memory**: The provided reference sources omit critical context, contradict the full conversation memory, or indicate a resolution of the task that does not match the ground-truth or the question as posed.

### Ground-truth answer

This category addresses errors in the canonical answer string(s), where the supposed correct answer is itself wrong or ill-defined relative to the memory-backed task.

* **Ungrounded or over-specific ground-truth**: The canonical answer asserts entities, quantities, dates, or other specifics that are not supported (or are contradicted) by the memory and reference material.
* **Incompatible listed ground-truth variants**: Multiple acceptable ground-truth strings are listed, but they are mutually incompatible, or none align with the memory-backed solution.

## Crucial evaluation rules

* You **must** read both the **reference sources** and the **full conversation memory** before deciding. The reference excerpt may be partial; the full memory is authoritative for what the agent could recall.
* Superficial string differences that normalization would reconcile are **not** flaws.
* Flag a sample as flawed **only** when a perfect agent with faithful access to the given memory could not reasonably justify the ground-truth answer. Do **not** mark flawed merely because the answer required light paraphrase or minor inference that a careful reader would accept.

## Evaluation and Output Format

Carefully analyze the provided sample. Think step-by-step and map any flaw to **one primary category** from the taxonomy above (the single best-matching leaf issue).

Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, name the flaw category and explain why it applies, citing the question, memory, or reference sources. If it is not flawed, briefly explain how the ground-truth is supported.",
  "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, state that it is not flawed. If flawed, state the issue concisely, e.g., The ground-truth date contradicts the date given earlier in the conversation memory.",
  "error_category": "The exact leaf-level issue name from the flaw categories above (e.g., \"Contradiction between ground-truth and memory\"). If the sample is not flawed, use \"Not Flawed\". If multiple issues apply, pick the single most severe or primary one.",
  "is_flawed": <true or false>
}}
```

## Sample to Evaluate

### Question
{instruction}

### Ground-Truth Answer
{ground_truth}

### Reference Sources
{sources}

### Conversation Memory
{memory_context}
"""

WEB_SEARCH_FILTERING_PROMPT = """
You are an expert evaluator for **Berkeley Function Calling Leaderboard (BFCL) V4 Agentic, Part 1: Web Search**, which examines an LLM's ability to use a web search API to answer knowledge-seeking questions that lie beyond its training data. This benchmark evaluates agents' ability to perform real-time web searches at the time questions are asked, meaning questions may reference time-sensitive information, future dates, or events that occur after the benchmark's creation date.
Your task is to **determine whether the given sample is fundamentally flawed**, meaning that even a perfect agent with unrestricted access to the internet could not reasonably solve the question **as the benchmark defines it**, given the evidence the benchmark supplies.

You will be provided with the following information:

* **Question**: The question that was asked to the LLM model. It often requires multi-hop reasoning and evidence retrieval through web search.
* **Ground-Truth Answer**: The canonical answer string(s) that must be contained (after normalization) in a model's output for it to be marked correct.
  Normalization converts text to lowercase and strips punctuation marks such as ,./-\\_*^() so that superficial differences (e.g., "eiffel-tower" vs "Eiffel Tower") do not affect scoring.
* **Reference Sources**: Example URLs or text passages that the benchmark used to construct or validate this question. These represent the type of information a real-time web search might find.

A sample is **flawed** if it exhibits one or more of the issues described below.

## Flaw Categories

Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.

### Question (task specification)

This category covers flaws in how the question is posed relative to what can be answered from web-style evidence.

* **Incoherent or unanswerable question**: The question is logically garbled, self-contradictory, or cannot be answered from any reasonable search workflow implied by the benchmark.
* **Underspecified or multiply-answerable question**: The question admits several materially different correct answers that typical search snippets could support, yet the benchmark commits to a single ground-truth without adequate grounding to prefer one over the others.

### Reference sources (as provided)

This category covers flaws in the supplied URLs or text passages **as given in the prompt**, relative to what the benchmark expects a search agent to recover. You are **not** asked to verify live web pages or real-world truth at adjudication time.

* **Insufficient evidence in provided references**: The excerpts (taken together) do not contain enough information for a diligent agent to justify the ground-truth answer, even with generous reading and ordinary multi-hop stitching.
* **Contradictory or incompatible reference material**: The provided passages jointly support a different answer than the ground-truth, or are internally inconsistent in a way that makes a single canonical answer ill-defined.

### Ground-truth answer

This category addresses errors in the canonical answer string(s) relative to the question and the provided reference material.

* **Ground-truth not deducible from references**: The canonical answer does not follow from the provided passages (wrong entity, wrong fact, wrong scope, or an over-specific answer with no support in the excerpts).
* **Ground-truth contradicts supported reading**: The canonical answer conflicts with what the provided references clearly support when read in good faith.

## Crucial evaluation rules

* **Evidence scope**: Judge only whether a reasonable model could deduce the ground-truth from the **reference sources included in this prompt**. Do **not** mark flawed because you cannot verify external sites or because your own knowledge disagrees with the excerpts.
* **Time-sensitive questions**: Agents search at query time; questions about future dates or evolving facts are **in scope** and are **not** inherently flawed.
* **Superficial differences**: Normalization handles minor formatting; those are **not** flaws.
* Flag a sample as flawed **only** when the mismatch is clear from the provided question, ground-truth, and reference material—not from speculation about unstated search results.

## Evaluation and Output Format

Carefully analyze the provided sample. Think step-by-step and map any flaw to **one primary category** from the taxonomy above (the single best-matching leaf issue).

Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, name the flaw category and explain why it applies, citing the question, reference sources, or ground-truth. If it is not flawed, briefly explain how the ground-truth is deducible from the references.",
  "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, state that it is not flawed. If flawed, state the issue concisely, e.g., The ground-truth entity name does not appear in any of the provided reference passages.",
  "error_category": "The exact leaf-level issue name from the flaw categories above (e.g., \"Insufficient evidence in provided references\"). If the sample is not flawed, use \"Not Flawed\". If multiple issues apply, pick the single most severe or primary one.",
  "is_flawed": <true or false>
}}
```

## Sample to Evaluate

### Question
{instruction}

### Ground-Truth Answer
{ground_truth}

### Reference Sources
{sources}
"""

_PROMPTS_BY_DOMAIN: Dict[str, str] = {
    "memory": MEMORY_FILTERING_PROMPT,
    "web_search": WEB_SEARCH_FILTERING_PROMPT,
}


def build_prompt(question: BFCLv4Question, step: LLMJudgeStep) -> str:
    if step != LLMJudgeStep.BENCHMARK_FILTER:
        raise ValueError(f"Unsupported judge step for BFCL v4: {step.value}")
    domain = _infer_domain(question)
    template = _PROMPTS_BY_DOMAIN.get(domain, DEFAULT_FILTERING_PROMPT)
    fields = _extract_format_fields(template)
    args = {_field: _render_field(question, _field) for _field in fields}
    return template.format(**args)


def _infer_domain(question: BFCLv4Question) -> str:
    task_name = (question.task_name or "").lower()
    if task_name:
        return task_name
    question_id = (question.question_id or "").lower()
    return question_id.split("_")[0] if question_id else ""


def _render_field(question: BFCLv4Question, field: str) -> str:
    if field == "memory_context":
        return _to_pretty_json(getattr(question, "memory_context", None))
    if field == "memory_source":
        return _to_pretty_json(getattr(question, "sources", None))
    if field == "research_trail":
        return _to_pretty_json(getattr(question, "sources", None))
    if field == "ground_truth":
        return _to_pretty_json(_get_value(question, "ground_truth"))
    if field == "gt_conv_traj":
        return _to_pretty_json(_get_value(question, "gt_conv_traj"))
    if field == "num_hops":
        return _as_text(_get_value(question, "num_hops"), "N/A")
    if field == "format_profile":
        return _as_text(_get_format_metadata(question, "config"), "N/A")
    if field == "base_category":
        return _as_text(_get_format_metadata(question, "base_category"), "N/A")
    if field == "question_turns":
        return _to_pretty_json(getattr(question, "question_turns", None))
    if field == "available_function_list":
        return _to_pretty_json(getattr(question, "available_function_list", []))

    value = _get_value(question, field)
    if isinstance(value, (dict, list)):
        return _to_pretty_json(value)
    return _as_text(value, "N/A")


def _get_value(question: BFCLv4Question, field: str):
    if hasattr(question, field):
        return getattr(question, field)
    meta = getattr(question, "meta", None)
    if isinstance(meta, dict) and field in meta:
        return meta[field]
    return None


def _get_format_metadata(question: BFCLv4Question, key: str):
    meta = getattr(question, "meta", None)
    if isinstance(meta, dict):
        fs_meta = meta.get("format_sensitivity", {})
        if isinstance(fs_meta, dict):
            return fs_meta.get(key)
    return None


def _extract_format_fields(template: str) -> List[str]:
    return re.findall(r"(?<!\{)\{([^{}]+)\}(?!\})", template)


def _to_pretty_json(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _as_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return _to_pretty_json(value)


def _sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[\\/:*?\"<>|]", "_", value)
    return sanitized.replace(" ", "_")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render BFCL v4 judge prompts")
    parser.add_argument(
        "-q",
        "--question-id",
        help="Specific BFCL v4 question ID to render",
    )
    parser.add_argument(
        "-s",
        "--step",
        choices=[step.value for step in LLMJudgeStep],
        default=LLMJudgeStep.BENCHMARK_FILTER.value,
        help="LLM judge step to render (default: benchmark_filter)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Destination file for the rendered prompt (single-question mode)",
    )
    parser.add_argument(
        "--output-dir",
        default="bfcl_v4_prompts",
        help="Directory to store prompts when exporting the full set",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Export prompts for every BFCL v4 question",
    )
    return parser.parse_args()


def _load_questions() -> List[BFCLv4Question]:
    from src.bench_loaders.bfcl_v4_loader import BfclV4Loader

    loader = BfclV4Loader()
    return loader.load_questions()


def _render_prompt(question: BFCLv4Question, step: LLMJudgeStep) -> str:
    return build_prompt(question, step)


def _write_text(path: str, content: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _export_single_prompt(
    question: BFCLv4Question,
    step: LLMJudgeStep,
    output_path: str,
) -> None:
    prompt_text = _render_prompt(question, step)
    _write_text(output_path, prompt_text)
    print(f"Saved {step.value} prompt for {question.question_id} to {output_path}")


def _export_all_prompts(
    questions: List[BFCLv4Question],
    step: LLMJudgeStep,
    base_dir: str,
) -> None:
    total_written = 0
    for question in questions:
        domain = (question.task_name or "unknown_domain").lower()
        filename = f"{_sanitize_filename(question.question_id)}__{step.value}.txt"
        output_path = os.path.join(base_dir, step.value, domain, filename)
        prompt_text = _render_prompt(question, step)
        _write_text(output_path, prompt_text)
        total_written += 1

    print(
        f"Saved {total_written} prompts for step '{step.value}' under {os.path.join(base_dir, step.value)}"
    )


def _select_question(
    questions: List[BFCLv4Question],
    question_id: str,
) -> BFCLv4Question:
    for question in questions:
        if question.question_id == question_id:
            return question
    raise ValueError(f"Could not find BFCL v4 question with ID '{question_id}'")


def _default_output_filename(step: LLMJudgeStep) -> str:
    return f"bfcl_v4_{step.value}_prompt.txt"


def _main() -> None:
    args = _parse_args()
    step = LLMJudgeStep(args.step)
    questions = _load_questions()

    if args.save_all:
        _export_all_prompts(questions, step, args.output_dir)
        return

    if not args.question_id:
        raise SystemExit("Error: --question-id is required when not using --save-all")

    try:
        selected = _select_question(questions, args.question_id)
    except ValueError as exc:
        raise SystemExit(str(exc))

    output_path = args.output or _default_output_filename(step)
    _export_single_prompt(selected, step, output_path)
