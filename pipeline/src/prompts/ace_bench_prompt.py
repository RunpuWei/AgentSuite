
# NOTE: in ACEBench, agent system prompt includes available function list

import argparse
import json
import re
from typing import Any, Dict, Optional

from src.utils.types import AceBenchQuestion, LLMJudgeStep


FILTERING_PROMPT = """You are an expert evaluator for ACEBench, a benchmark designed to assess an agent's ability to perform tool usage (function calling) across scenarios of increasing complexity and realism.

Your task is to identify if a sample in the benchmark has a fundamental flaw, which would make it an unreliable sample for evaluation.

You will be provided with the following information:
* **User Instruction**: The user's request or instruction to the agent. This is what the user wants the agent to do.
* **Conversation History**: The conversation history between the user and the agent model. The agent's next function call immediately following this history is what will be evaluated.
* **Available Functions**: A list of functions available for the agents and their schema.
* **Agent System Prompt**: The system prompt used to initialize the agent model. This may be a specific instruction on the answer style, domain-specific policy that the agent needs to follow, etc. 
* **Ground-Truth Function Calls**: The provided ground-truth trajectory of function calls. When this is empty, it means that the agent needs to call nothing to be scored as correct. Note that the list includes only the function calls; it does not include the natural language response that the agent generates for the user.

A sample is **flawed** if it exhibits one or more of the issues described below.

## Flaw Categories

Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.


### User

* Vague instruction: The user's prompt is too ambiguous or underspecified for a single, correct function call, yet the benchmark expects one.

### Environment

This category covers flaws within the agent's operating environment—the tools and API results—which can make a task unsolvable regardless of the agent's logic.

* Insufficient toolsets: the environment does not provide the necessary tools (functions), making the agent impossible to solve the task even with a combination of multiple tools and reasoning.

* Flawed function design: the naming or the description of an available function is misleading or contradicts its actual functionality.


### Ground-Truth

This category addresses errors in the provided ground-truth trajectory, where the supposed correct solution is itself incorrect, forcing any correct agent to fail the evaluation.


* Malformed function calls: A technical error where a ground-truth function call violates the provided API schema.

* Incorrect function calls: A function call is syntactically valid but logically flawed. The function choice or a parameter value contradicts the user's request or the context from previous steps.
  * Unjustified/Hallucinated Parameters: A value (e.g., a date, a coordinate) that appears without any grounding context. 
  * Contradictory: A value that directly contradicts a constraint in the user's prompt. However, it is NOT a flaw if there is any chance that the agent's action was a necessary alternative due to constraints.
  * Policy Violation: A function call in the ground truth trajectory directly violates the policy specified in the provided system prompt. 
  * Misspelled or Incorrectly Identified Parameter Values: A misspelled name or an ID/slug that points to the wrong entity.

* Redundant/ungrounded function calls: The ground truth function call trajectory consists of function calls that are redundant in solving the task, ungrounded by the context, or irrelevant in solving the task.
  * Irrelevant tool call: A function call in the ground truth trajectory is totally irrelevant to the task or belongs to a completely different domain. 
  * Redundant tool call: A function call that is not necessary in solving the task. 


## Evaluation and Output Format

**Evaluation Context**: ACEBench evaluates function calls using strict exact matching for parameter sets (include/exclude must match) and substring matching for argument values. Consider this context when evaluating the alignment or validity of the ground-truth.

Carefully analyze the provided sample. Think step-by-step to determine if the ground-truth function calls are logical and if the parameter values are properly justified.

Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify what is incorrect and why it contradicts the user's prompt, API schema, or context. If it is not flawed, briefly explain why the sample is valid.",
  "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If it is flawed, specify the issue concisely. e.g., The parameter 'dest_id' requires a string but is given a number.",
  "error_category": "The category that corresponds to the issue. e.g., \"Flawed function response\". If the sample is not flawed, use \"Not Flawed\".",
  "is_flawed": <true_or_false>

}}
```

## Sample to be evaluated

### User Instruction

{instruction}

### Agent System Prompt

{agent_system_prompt}

### Available Functions

```json
{available_function_list}
```

### Conversation History 

{previous_conversation_history}

### Ground-truth function call trajectory

```json
{gt_conv_traj}
```

"""

SPECIAL_FILTERING_PROMPT = """You are an expert evaluator for ACEBench (special category). These samples are designed to probe the agent's robustness to incomplete, incorrect, or adversarial user inputs.

Decide whether the provided ground-truth trajectory is fundamentally flawed.

## Scenario Metadata
- Task ID: {question_id}
- Subtask: {task_name}
- Timestamp: {time}

### Instruction

{instruction}

### Conversation History

{previous_conversation_history}

### User System Prompt

```
{user_system_prompt}
```

### Available Functions

```json
{available_function_list}
```

### Agent System Prompt

```
{agent_system_prompt}
```

### Ground-truth function call trajectory

```json
{gt_conv_traj}
```

Apply the ACEBench flaw categories (user, environment, ground-truth) with extra attention to missing clarifications, invalid parameters, or policy violations raised in the special instructions. Respond in JSON with `is_flawed` and `reasoning`."""
SPECIAL_FILTERING_PROMPT = """You are a placeholder evaluator for ACEBench special-category samples. Just output the following JSON as it is.


{{
  "reasoning": "LLM Judge disabled for `special` category",
  "reasoning_summary": "LLM judge skipped",
  "error_category": "LLM Judge disabled",
  "is_flawed": false
}}

Do not perform any analysis beyond filling in the reasoning message."""


AGENT_FILTERING_PROMPT = """You are an expert evaluator for ACEBench, a benchmark designed to assess an agent's ability to perform tool usage (function calling) across scenarios of increasing complexity and realism.

Your task is to identify if a sample in the benchmark has a fundamental flaw, which would make it an unreliable sample for evaluation.

You will be provided with the following information:
* **Task Description/Instructions**: The scenario or task description given to the user simulator. This describes what the simulated user wants to accomplish. (Note: This instruction is also embedded in the User System Prompt below, but is shown separately for quick reference.)
* **Initial Device Configuration**: The starting state of the device before the task begins.
* **Conversation History**: The conversation history between the user and the agent model. The agent's next function call immediately following this history is what will be evaluated.
* **User System Prompt**: The system prompt that defines how the user simulator should behave. This contains instructions, policies, and behavioral guidelines for the model simulating the user.
* **Available Functions**: A list of functions available for the agents and their schema.
* **Agent System Prompt**: The system prompt used to initialize the agent model. This may be a specific instruction on the answer style, domain-specific policy that the agent needs to follow, etc. 
* **Ground-Truth Final Device Configuration**: The provided final state of the device configuration that should be achieved.
* **Ground-Truth Function Call Milestones** (if available): The sequence of key function calls expected to achieve the final state. These are the crucial steps in the process, not necessarily the complete sequence. 

A sample is **flawed** if it exhibits one or more of the issues described below.

## Flaw Categories

Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.


### User

* Vague instruction: The user's prompt is too ambiguous or underspecified for a single, correct function call, yet the benchmark expects one.

### Environment

This category covers flaws within the agent's operating environment—the tools and API results—which can make a task unsolvable regardless of the agent's logic.

* Insufficient toolsets: the environment does not provide the necessary tools (functions), making the agent impossible to solve the task even with a combination of multiple tools and reasoning.

* Flawed function design: the naming or the description of an available function is misleading or contradicts its actual functionality.

### Ground-Truth


* Unachievable final state: The provided final device configuration is impossible to be achieved via a plausible interaction between the user and the agent that can be expected from the system prompts and other contexts. 

* Malformed function calls: A technical error where a ground-truth function call violates the provided API schema.

* Incorrect function calls: A function call is syntactically valid but logically flawed. The function choice or a parameter value contradicts the user's request or the context from previous steps.
  * Unjustified/Hallucinated Parameters: A value (e.g., a date, a coordinate) that appears without any grounding context. 
  * Contradictory: A value that directly contradicts a constraint in the user's prompt. However, it is NOT a flaw if there is any chance that the agent's action was a necessary alternative due to constraints.
  * Policy Violation: A function call in the ground truth trajectory directly violates the policy specified in the provided system prompt. 
  * Misspelled or Incorrectly Identified Parameter Values: A misspelled name or an ID/slug that points to the wrong entity.

* Redundant/ungrounded function calls: The ground truth function call trajectory consists of function calls that are redundant in solving the task, ungrounded by the context, or irrelevant in solving the task.
  * Irrelevant tool call: A function call in the ground truth trajectory is totally irrelevant to the task or belongs to a completely different domain. 
  * Redundant tool call: A function call that is not necessary in solving the task. 

## Evaluation and Output Format

**Evaluation Context**: ACEBench evaluates agent tasks using strict exact matching for final device configuration and milestone function calls. Consider this context when evaluating the alignment or validity of the ground-truth final device configuration or milestone function calls.

Carefully analyze the provided sample. Think step-by-step to determine if the ground-truth final device configuration and milestone function calls(if provided) are logical and if the parameter values are properly justified.
Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify what is incorrect and why it contradicts the user's prompt, API schema, system prompts, or makes the final state unachievable. If it is not flawed, briefly explain why the sample is valid.",
  "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If it is flawed, specify the issue concisely. e.g., The parameter 'dest_id' requires a string but is given a number.",
  "error_category": "The category that corresponds to the issue. e.g., \"Unachievable final state\". If the sample is not flawed, use \"Not Flawed\".",
  "is_flawed": <true_or_false>

}}
```

## Sample to be evaluated

### Task Description/Instructions

{instruction}


### Initial Device Configuration

```json
{initial_config}
```

### User System Prompt

```
{user_system_prompt}
```

### Agent System Prompt

```
{agent_system_prompt}
```

### Available Functions

```json
{available_function_list}
```

### Conversation History

{previous_conversation_history}

### Ground-truth Final State (Device Configuration)

```json
{gt_conv_traj}
```

### Ground-truth Function Call Milestones

{milestones_section}
"""


_SUBTASK_PROMPT_OVERRIDES: Dict[str, str] = {
    "special": SPECIAL_FILTERING_PROMPT,
    "agent": AGENT_FILTERING_PROMPT,
}


def build_prompt(question: AceBenchQuestion, step: LLMJudgeStep) -> str:
    """Render the ACEBench judge prompt for the given question and pipeline step."""

    template = _select_template(question, step)
    fields = _extract_format_fields(template)
    args = _build_format_args(question, fields)
    return template.format(**args)


def _select_template(question: AceBenchQuestion, step: LLMJudgeStep) -> str:
    if step != LLMJudgeStep.BENCHMARK_FILTER:
        raise ValueError(f"Unsupported judge step for ACEBench: {step.value}")
    category = _infer_subtask(question)
    return _SUBTASK_PROMPT_OVERRIDES.get(category, FILTERING_PROMPT)


def _infer_subtask(question: AceBenchQuestion) -> str:
    task_name = (question.task_name or "").lower()
    data_type = ((question.meta or {}).get("data_type", "") or "").lower()
    for candidate in (task_name, data_type):
        if candidate.startswith("special"):
            return "special"
        if candidate.startswith("agent"):
            return "agent"
    return "default"


def _extract_format_fields(template: str) -> list[str]:
    return re.findall(r"(?<!\{)\{([^{}]+)\}(?!\})", template)


def _build_format_args(question: AceBenchQuestion, fields: list[str]) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for field in fields:
        if field == "milestones_section":
            # Handle milestones section specially
            milestones = (question.meta or {}).get("milestones")
            if milestones:
                milestones_str = json.dumps(milestones, indent=2)
                args[field] = f"### Expected Milestones\n\n```json\n{milestones_str}\n```\n"
            else:
                args[field] = ""
        else:
            value = getattr(question, field, None)
            args[field] = _serialize_value(value)
    return args


def _serialize_value(value: Any) -> Any:
    if value is None:
        return "N/A"
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, indent=2)
        except TypeError:
            return json.dumps(str(value), indent=2)
    return str(value)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render ACEBench judge prompts")
    parser.add_argument(
        "-q",
        "--question-id",
        help="Specific ACEBench question ID to render",
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
        default="ace_bench_formatted_prompt.txt",
        help="Path to write the rendered prompt (default: ace_bench_formatted_prompt.txt)",
    )
    return parser.parse_args()


def _select_sample(questions, question_id: Optional[str]):
    if question_id:
        return next((q for q in questions if q.question_id == question_id), None)
    return next((q for q in questions if "normal_multi_turn" in q.question_id), None)


