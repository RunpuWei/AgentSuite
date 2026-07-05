FILTERING_PROMPT = """
You are an expert evaluator for **NexusBench**, a benchmark designed to assess precise tool-use across diverse tool-use tasks.
Your task is to determine if a given benchmark sample has a fundamental flaw in its user prompt, environment, or ground-truths, which would make it unable to be incorporated in the evaluation.

You will be provided with the following context for each sample:
* **System Prompt**: The system-level instructions supplied to the agent model.
* **User Prompt**: The user's task description and instructions.
* **Ground Truth Output**: The benchmark’s reference answer. This can be either single or a chain of function calls, or final output expected.
* **Sub-benchmark Name**: The specific NexusBench sub-benchmark the sample belongs to.
* **Sub-benchmark Description**: The detailed description of the sub-benchmark the sample belongs.
* **Tool Schemas** – complete JSON definitions for every callable tool.


A sample is **flawed** if it exhibits one or more of the issues described below.

## Flaw Categories

Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.

### User

* Vague instruction: The user's prompt is too ambiguous or underspecified for a single, correct function call, yet the benchmark expects one.
  * However, ONLY flag this if the user query cannot be reasonably inferred from the available tools and system instructions. If you are unsure, flag it as valid. 


### Environment

This category covers flaws within the agent's operating environment—the tools and API results—which can make a task unsolvable regardless of the agent's logic.

* Insufficient toolsets: the environment does not provide the necessary tools (functions), making the agent impossible to solve the task even with a combination of multiple tools and reasoning.

* Flawed function design: the naming or the description of an available function is misleading or contradicts its actual functionality.


### Ground-Truth

This category covers flaws in the pre-defined ground truth of the sample. As described previously, the ground truth can be a single function call, a chain of function calls, or a numerical value that is the outcome of the ground truth function calls. If the ground truth is a single number, ignore this error type.

* Malformed function calls: A technical error where a ground-truth function call violates the provided API schema.

* Incorrect function calls: A function call is syntactically valid but logically flawed. The function choice or a parameter value contradicts the user's request or the context from previous steps.
  * Unjustified/Hallucinated Parameters: A value (e.g., a date, a coordinate) that appears without any grounding context. For example, searching for a hotel on a date that was not returned by a preceding flight search.
  * Contradictory Parameter Values: A value that directly contradicts a constraint in the user's prompt. For example, computing a date difference as 19*365 + 23 when the user requested “19 years and 23 days ago,” without accounting for leap years
  * Misspelled or Incorrectly Identified Parameter Values: A misspelled name or an ID/slug that points to the wrong entity (e.g., selecting the wrong airport ID).


* Redundant/ungrounded function calls: The ground truth function call trajectory consists of function calls that are redundant in solving the task, ungrounded by the context, or irrelevant in solving the task.
  * Irrelevant tool call: A function call in the ground truth trajectory is totally irrelevant to the task or belongs to a completely different domain. Example: agent calls a function to reserve a flight, though it was asked to process product exchange.
  * Redundant tool call: A function call that is not necessary in solving the task. Example: the agent is asked to search for attractions until it finds one that meets a certain condition; However, the agent performs the search in an arbitrary order, resulting in an excessive number of function calls.

## Crucial Note: Sub-benchmark-specific note for TMIHallucination
  
Some samples in TMIHallucination deliberately ship tool schemas where almost every parameter is flagged as `required`. That alone is not a flaw—those schemas are meant to pressure-test whether the ground truth omits the explicitly banned argument. Treat the usual NexusBench rules as primary; do **not** mark a sample flawed solely because a schema lists an excessive number of required fields, or because the ground-truth call omits required fields. Ignore this note if the sample belongs to other sub-benchmarks.


## Evaluation and Output Format
Carefully analyze the provided sample. Think step-by-step to determine if the ground-truth trajectory is a correct and logical solution to the user's prompt.

Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the ground-truth is flawed, specify which argument is incorrect and why it contradicts the prompt or schema. If it is not flawed, briefly explain why the ground-truth is a correct interpretation of the user's request.",
  "reasoning_summary": "A shorter rationale for your decision. If the ground-truth is not flawed, just mention that it is not flawed. If the ground-truth is flawed, specify the issue concisely. e.g., The argument `search_type` in the function call `Search_Hotels` is supposed to be `district`, but is misspelled as `dustrict`.",
  "error_category": "The category that corresponds to the issue. e.g., \"Flawed function response\". If the sample is not flawed, use \"Not Flawed\".",
  "is_flawed": <true_or_false>
}}
```


## Target Sample

### Sub-benchmark Description

{subbench_description}

### System Prompt
```
{system_prompts}
```

### User Prompt
```
{instruction}
```

### Ground Truth Output
```
{reference}
```

### Available Tools (JSON Schemas):
```
{tool_definitions}
```

{tool_implementations}
"""


