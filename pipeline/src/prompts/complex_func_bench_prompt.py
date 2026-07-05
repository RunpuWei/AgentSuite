FILTERING_PROMPT = """
You are an expert evaluator for ComplexFuncBench, a benchmark designed to assess the complex function-calling capabilities of LLMs. 
Your task is to determine if a given benchmark sample has a fundamental flaw in its user prompt, environment, or ground-truths, which would make it unable to be incorporated in the evaluation.

You will be provided with:

* **User Prompt:** The user's task description and instructions.
* **Available Functions:** A complete list of functions the agent can use, including their JSON schemas.
* **Ground-Truth Trajectory:** The reference sequence of function calls. Note that messages with `"role": "observation"` are the outputs of the immediately preceding function call. 
  
A sample is **flawed** if it exhibits one or more of the issues described below.

## Flaw Categories

Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.

### User

* Vague instruction: The user's prompt is too ambiguous or underspecified for a single, correct function call, yet the benchmark expects one.


### Environment

This category covers flaws within the agent's operating environment—the tools and API results—which can make a task unsolvable regardless of the agent's logic.

* Flawed function response: The pre-computed API response provided in the benchmark is incorrect, misleading, or doesn't contain the information needed to fulfill the user's request. Since the agent relies on these responses, a flawed API response makes the task unsolvable.
  * Look for:
    * Incorrect resolution: An ambiguous name in the function call is resolved to the wrong entity in the response.
    * Irrelevant results: The API returns a list of items that are completely irrelevant to the user's request 

* Insufficient toolsets: the environment does not provide the necessary tools (functions), making the agent impossible to solve the task even with a combination of multiple tools and reasoning.

* Flawed function design: the naming or the description of an available function is misleading or contradicts its actual functionality.


### Ground-Truth

This category addresses errors in the provided ground-truth trajectory, where the supposed correct solution is itself incorrect, forcing any correct agent to fail the evaluation.


* Malformed function calls: A technical error where a ground-truth function call violates the provided API schema.

* Incorrect function calls: A function call is syntactically valid but logically flawed. The function choice or a parameter value contradicts the user's request or the context from previous steps.
  * Unjustified/Hallucinated Parameters: A value (e.g., a date, a coordinate) that appears without any grounding context. 
  * Contradictory Parameter Values: A value that directly contradicts a constraint in the user's prompt. 
  * Misspelled or Incorrectly Identified Parameter Values: A misspelled name or an ID/slug that points to the wrong entity.

* Redundant/ungrounded function calls: The ground truth function call trajectory consists of function calls that are redundant in solving the task, ungrounded by the context, or irrelevant in solving the task.
  * Irrelevant tool call: A function call in the ground truth trajectory is totally irrelevant to the task or belongs to a completely different domain. 
  * Redundant tool call: A function call that is not necessary in solving the task. 

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


## Sample to be evaluated

### User's Prompt

```
{instruction}
```

### List of available functions and their schema

```json
{available_function_list}
```

### Ground-truth function call trajectory 
* Note that messages with "role": "observation" are the results of the function call right before.

```json
{gt_conv_traj}
```
"""

