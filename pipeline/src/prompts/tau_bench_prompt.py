# Please fill in the prompts to resolve the identified issue. You can refer to the prompt in src/prompts/complex_func_bench_prompt.py
# Make sure the output format is as follows. Beware the `reasoning` attribute needs to preceed the results (`is_flawed` or `score`) to encourage the model's chain-of-thought reasoning.


FILTERING_PROMPT = """
You are an expert evaluator for Tau-Bench, a benchmark designed to assess an agent's ability to follow complex rules and interact with a simulated user.
Your task is to determine if a given benchmark sample has a fundamental flaw in its user prompt, environment, or ground-truths, which would make it unable to be incorporated in the evaluation.


You will be provided with the following information:
* **User Scenario**: The system prompt given to the model that simulates user. You need to expect how the user model would behave based on this.
* **System Policy**: Domain-specific rules that the agent model needs to obey. This will be given as the system prompt for agent models.
* **Available Function List**: a list of functions available for the agents and their schema
* **User Context and Relevant Information**: a brief information of the user and relevant information.
* **Ground-Truth Milestone Function Calls**: the provided ground-truth trajectory. Note that this is not a complete log of all function calls. Instead, it is a curated list containing only the key milestone function calls required to solve the task. Note that messages with `"role": "observation"` are the outputs of the immediately preceding function call. 

A sample is **flawed** if it exhibits one or more of the issues described below.

## Flaw Categories

Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.

### Environment

This category covers flaws within the agent's operating environment—the tools and API results—which can make a task unsolvable regardless of the agent's logic.

* Flawed function design: the naming or the description of an available function is misleading or contradicts its actual functionality.
  * Example: A function named `vt_get_votes_on_ip_address` provides "example.com" as an example for its argument value in its schema. 

### Ground-Truth

This category addresses errors in the provided ground-truth trajectory, where the supposed correct solution is itself incorrect, forcing any correct agent to fail the evaluation.


* Malformed function calls: A technical error where a ground-truth function call violates the provided API schema.
  * Example: A parameter requires a string but is given a number (e.g., dest_id: 123 instead of dest_id: "123"), a required parameter is missing, the function name is wrong, or a parameter value is misspelled (e.g., sort_by: "popularitye" instead of "popularity").

* Incorrect function calls: A function call is syntactically valid but logically flawed. The function choice or a parameter value contradicts the user's request or the context from previous steps.
  * Unjustified/Hallucinated Parameters: A value (e.g., a date, a coordinate) that appears without any grounding context. For example, searching for a hotel on a date that was not returned by a preceding flight search.
  * Contradictory: A value that directly contradicts a constraint in the user's prompt. However, it is NOT a flaw if there is any chance that the agent's action was a necessary alternative due to constraints like an insufficient budget or a lack of available seats.
  * Policy Violation: A function call in the ground truth trajectory directly violates the provided system policy. Example: The ground truth where the agent calls a specific function twice, although it is mentioned in the system policy that the function can only be called once.
  * Misspelled or Incorrectly Identified Parameter Values: A misspelled name or an ID/slug that points to the wrong entity (e.g., selecting the wrong airport ID).

* Redundant/ungrounded function calls: The ground truth function call trajectory consists of function calls that are redundant in solving the task, ungrounded by the context, or irrelevant in solving the task.
  * Irrelevant tool call: A function call in the ground truth trajectory is totally irrelevant to the task or belongs to a completely different domain. Example: agent calls a function to reserve a flight, though it was asked to process product exchange.
  * Redundant tool call: A function call that is not necessary in solving the task. However, note that function calls that does not change the database status are tolerable even if they are redundant. EXAMPLE: a sample is not flawed even if calls `get_order_details` for irrelevant items. 


## Crucial Rule: Actively Reconstruct the Conversation

The ground-truth trajectory only contains key milestone function calls. It intentionally omits function calls that are less important for evaluation and the natural language conversation between the user and the agent (e.g., user confirmations, request, clarifications, or follow-up questions).
Your task is to find undeniable flaws. Therefore, you MUST operate under the following assumption:

* For example, the ground truth milestone sequence may not contain a call that authenticates the user identity. It may have been intentionallly omitted from the milestone sequence, since it is considered less important than calls that explicitly process user requests. Therefore, lack of authentication, user's confirmation or request, clarification should NOT be the sole reason to judge a sample as flawed.
* If a sequence of function calls can be justified by a plausible, un-shown conversation that does not contradict the User Scenario or System Policy, then it is NOT a flaw. The agent would have explained the user why it cannot process his request, although it is not shown in the milestone trajectory.
* In other words, imagine a possible conversation history that would justify the ground truth milestone function call trajectory. When you contemplate of a plausible trajectory, note that the user can make a request that is not mentioned in the prompt, guided by the agent. Flag a sample as flawed ONLY if a function call is impossible to justify, even with a hypothetical conversation. Do NOT infer a flaw from missing conversational steps.


## Evaluation and Output Format
Carefully analyze the provided sample. Think step-by-step to determine if the ground-truth trajectory is a correct and logical solution to the user's prompt.

Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify what is incorrect and why it contradicts the user's prompt, system policies, or the user's role. If it is not flawed, briefly explain why the sample is valid.",
  "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If it is flawed, specify the issue concisely. e.g., The ground truth books a connecting flight, but the user requested a direct flight.",
  "error_category": "The category that corresponds to the issue. e.g., \"Incorrect function calls\". If the sample is not flawed, use \"Not Flawed\".",
  "is_flawed": <true or false>
}}
```


## Target Sample

### User Scenario

```
{instruction}
```

### System Policy

{agent_system_prompt}


### User context and relevant information
{user_context}

### List of available functions and their schema

```json
{available_function_list}
```


### Ground-Truth Milestone Function Calls 
* Note that messages with "role": "observation" are the results of the function call right before.

```json
{gt_conv_traj}
```
"""
