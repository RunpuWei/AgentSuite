from typing import Any
import os
import sys
from openai import OpenAI
import json
import re
import copy
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.prompts import SimpleTemplatePrompt
from utils.utils import *
from collections import defaultdict


class GPTModel:
    def __init__(self, model_name, is_user=False):
        super().__init__()
        self.model_name = model_name
        if is_user:
            self.client = OpenAI(
                api_key=os.getenv("API_KEY"),
                base_url=os.getenv("BASE_URL")
            )
        else:
            self.client = OpenAI(
                api_key=os.getenv("API_KEY"),
                base_url=os.getenv("BASE_URL")
            )
        

    def __call__(self, prefix, prompt: SimpleTemplatePrompt, **kwargs: Any):
        filled_prompt = prompt(**kwargs)
        prediction = self._predict(prefix, filled_prompt, **kwargs)
        return prediction
    
    @retry(max_attempts=10)
    def _predict(self, prefix, text, **kwargs):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                )
            return completion.choices[0].message.content
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['context length', 'token limit', 'maximum context', 'too many tokens']):
                print(f"Context length error: {e}", file=sys.stderr)
                return {"error_type": "context_length_exceeded", "error_message": str(e)}
            else:
                print(f"Exception: {e}", file=sys.stderr)
                return None


class FunctionCallGPT(GPTModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        
        self.messages = []

    @retry(max_attempts=5, delay=10)
    def __call__(self, messages, tools=None, **kwargs: Any):
        is_claude = ("claude" in self.model_name)
        is_thinking = ("thinking-on" in self.model_name)
        if "function_call" not in json.dumps(messages, ensure_ascii=False):
            self.messages = copy.deepcopy(messages)
        try:
            model_name = self.model_name
            kwargs = {
                "model": model_name,
                "messages": self.messages,
                "tools": tools,
            }

            # Reasoning models (gpt-5*, OpenAI o-series) must NOT get a tight
            # max_tokens cap: their reasoning tokens count against the completion
            # budget, so a 16384 cap truncates the actual answer to empty (this
            # silently sank o3/o4-mini: 65-81% empty completions). Treat them like
            # gpt-5 — auto tool choice, no max_tokens cap, no temperature.
            base_name = model_name.split("/")[-1]
            if re.search(r"gpt-5(?!\d)", model_name) or re.match(r"o[1-9]", base_name):
                kwargs["tool_choice"] = "auto"
            elif "gemini" in model_name:
                pass    # no additional args
            else:
                kwargs["temperature"] = float(os.getenv("AGENTSUITE_TEMPERATURE", 0.0))
                kwargs["tool_choice"]={"type": "auto"} if is_claude else "auto"
                kwargs["max_tokens"]=16384
                # Only set extra_body when actually enabling thinking; some
                # OpenAI-compatible proxies reject an explicit null extra_body.
                if is_thinking:
                    kwargs["extra_body"] = {
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": 16384
                        }
                    }

            completion = self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['context length', 'token limit', 'maximum context', 'too many tokens']):
                print(f"Context length error: {e}", file=sys.stderr)
                return {"error_type": "context_length_exceeded", "error_message": str(e)}
            else:
                print(f"Exception: {e}", file=sys.stderr)
                return None


if __name__ == "__main__":
    model = GPTModel("gpt-4")
    response = model("You are a helpful assistant.", SimpleTemplatePrompt(template=("What is the capital of France?"), args_order=[]))
    print(response)