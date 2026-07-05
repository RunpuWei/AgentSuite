"""ACEBench duplicate question filtering."""

from typing import Dict, List, Tuple
from .base_filter import BaseBenchmarkFilter
import logging
import json

logger = logging.getLogger(__name__)


class ACEBenchFilter(BaseBenchmarkFilter):
    """Remove duplicate ACEBench questions while leaving other samples untouched."""

    def __init__(self):
        super().__init__("ACEBench")

    def get_filter_name(self) -> str:
        return "ACEBench Duplicate Question Filter"

    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        logger.info(
            f"Applying ACEBench duplicate question filtering to {len(samples)} samples"
        )

        seen_models_by_question: Dict[str, set] = {}
        passed_samples: List[Dict] = []
        dropped_samples: List[Dict] = []

        for sample in samples:
            question_key = self._build_question_key(sample)
            model_key = self._extract_model_key(sample)

            models_for_question = seen_models_by_question.setdefault(question_key, set())
            if model_key in models_for_question:
                dropped_samples.append(sample)
                continue

            models_for_question.add(model_key)
            passed_samples.append(sample)

        self.log_filtering_stats(
            len(samples), len(passed_samples), len(dropped_samples)
        )
        return passed_samples, dropped_samples

    def _build_question_key(self, sample: Dict) -> str:
        # Extract initial_config first (critical for agent tasks)
        initial_config = sample.get("initial_config")
        config_key = ""
        if initial_config:
            # Normalize the config by converting to sorted JSON string
            try:
                config_key = json.dumps(initial_config, sort_keys=True, ensure_ascii=False)
            except (TypeError, ValueError):
                config_key = str(initial_config)
        
        # For agent tasks (agent_multi_turn, agent_multi_step), question_id is the primary identifier
        # Check if this is an agent task
        task_name = sample.get("task_name", "")
        is_agent_task = isinstance(task_name, str) and task_name.startswith("agent")
        
        # Try to get question_id from multiple possible locations
        # First try top-level
        question_id = sample.get("question_id")
        # Then try meta dict
        if not question_id:
            meta = sample.get("meta")
            if isinstance(meta, dict):
                question_id = meta.get("id") or meta.get("question_id")
        
        # For agent tasks, prioritize question_id as the primary identifier
        # This ensures different agent tasks (e.g., agent_multi_turn_0, agent_multi_turn_1) 
        # with same task_name and initial_config are not treated as duplicates
        if is_agent_task and question_id:
            if config_key:
                return f"{question_id}||{config_key}"
            return str(question_id)
        
        # For non-agent tasks or if question_id not available, use the original logic
        prompt_key = self._extract_prompt_key(sample)
        
        if prompt_key:
            if task_name:
                normalised_task = self._normalise_text(str(task_name))
                if normalised_task:
                    # Include config_key if present
                    if config_key:
                        return f"{normalised_task}||{prompt_key}||{config_key}"
                    return f"{normalised_task}||{prompt_key}"
            # Include config_key even without task_name
            if config_key:
                return f"{prompt_key}||{config_key}"
            return prompt_key

        # If we have question_id but it's not an agent task, use it
        if question_id:
            if config_key:
                return f"{question_id}||{config_key}"
            return str(question_id)

        instruction = sample.get("instruction")
        if instruction:
            instruction_key = self._normalise_text(str(instruction))
            if config_key:
                return f"{instruction_key}||{config_key}"
            return instruction_key

        # Fallback: include config in full JSON dump
        return json.dumps(sample, sort_keys=True, ensure_ascii=False)

    def _extract_prompt_key(self, sample: Dict) -> str:
        fragments: List[str] = []

        question_text = sample.get("question")
        if question_text:
            fragments.append(self._normalise_text(str(question_text)))

        messages = sample.get("messages")
        if isinstance(messages, list):
            fragments.extend(self._extract_messages(messages))

        conversation = sample.get("conversation")
        if isinstance(conversation, list):
            fragments.extend(self._extract_messages(conversation))

        if not fragments:
            return ""

        joined = " ".join(fragment for fragment in fragments if fragment)
        return self._normalise_text(joined)

    def _extract_messages(self, messages: List[Dict]) -> List[str]:
        fragments: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role not in {"system", "user"}:
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                fragments.append(self._normalise_text(content))
        return fragments

    def _extract_model_key(self, sample: Dict) -> str:
        model_name = sample.get("model_name")
        if model_name:
            return str(model_name)

        response = sample.get("response")
        if isinstance(response, dict):
            model = response.get("model") or response.get("model_name")
            if model:
                return str(model)

        return "__unknown_model__"

    def _normalise_text(self, text: str) -> str:
        return " ".join(text.split())
