"""
TAU Bench-specific rule-based filtering.
Implements custom filtering logic for TAU Bench evaluation data.
"""

from typing import Dict, List, Tuple
from .base_filter import BaseBenchmarkFilter
import logging
import numpy as np
import re
from collections import defaultdict
from src.bench_loaders.tau2_bench_loader import Tau2BenchLoader

logger = logging.getLogger(__name__)

class TAU2BenchFilter(BaseBenchmarkFilter):
    """TAU2 Bench-specific filtering rules."""
    
    def __init__(self):
        super().__init__("TAU2 Bench")
    
    def get_filter_name(self) -> str:
        return "TAU2 Bench-Specific Filter"
    
    def is_applicable(self, sample: Dict) -> bool:
        """Samples are already partitioned by benchmark before this filter runs."""
        return True
    
    def _normalize_question_id(self, task_name: str, meta_id: str) -> str:
        """
        Normalize question ID to handle inconsistent formats in meta['id'].
        Standardizes to underscore format (domain_id) to match pipeline conventions.
        For telecom, meta['id'] already includes the domain prefix (e.g., "telecom_[...]")
        For airline/retail, meta['id'] is usually just a number, but sometimes already formatted.
        """
        # Check if meta_id already starts with task_name (with underscore or hyphen)
        if meta_id.startswith(f"{task_name}_"):
            return meta_id  # Already in correct underscore format
        elif meta_id.startswith(f"{task_name}-"):
            # Convert hyphen format to underscore format for consistency
            return meta_id.replace(f"{task_name}-", f"{task_name}_", 1)
        else:
            # Format as domain_id (underscore) to match pipeline conventions
            return f"{task_name}_{meta_id}"
    
    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply TAU2 Bench-specific filtering rules.
        Note: Comprehensive filtering has already been applied before this method is called.
        """
        logger.info(f"Applying TAU2 Bench-specific filtering to {len(samples)} samples")
        
        # STEP 1: Filter questions solvable by a trivial agent, but keep those with success rate <= 0.5
        DO_NOTHING_SUCCESS_RATE_THRESHOLD = 0.5
        do_nothing_qids_to_filter = self._get_qids_solvable_by_do_nothing()

        # STEP 2: Filter questions with a vague communication information
        vague_qids_to_filter = self._get_qids_with_vague_communication_info()

        # STEP 3: Filter questions with high role confusion rate 
        ROLE_CONFUSION_FREQUENCY_THRESHOLD = 0.3
        role_confusion_qids_to_filter = self._get_qids_with_role_confusion(samples, ROLE_CONFUSION_FREQUENCY_THRESHOLD)
        logger.info(
            "%s question ids flagged for role-confusion filtering",
            len(role_confusion_qids_to_filter),
        )

        question_groups = self._group_samples_by_question(samples)
        passed_samples = []
        dropped_samples = []
        
        # Track filtering reasons for each qid
        qid_filter_reasons = defaultdict(list)
        # Track do_nothing_keep qids (for logging, but these are kept, not filtered)
        do_nothing_keep_qids = []

        for sample in samples:
            qid = self._normalize_question_id(sample['task_name'], sample['meta']['id'])
            mean_score = self._calculate_mean_score(question_groups[qid])
            
            filter_reasons = []
            
            # Check each filtering condition and track the reason
            if qid in do_nothing_qids_to_filter and mean_score is not None and mean_score > DO_NOTHING_SUCCESS_RATE_THRESHOLD:
                filter_reasons.append("do_nothing_high_success")
            elif qid in do_nothing_qids_to_filter:
                # Track for logging but don't filter out (these are kept)
                if qid not in do_nothing_keep_qids:
                    do_nothing_keep_qids.append(qid)
            if qid in role_confusion_qids_to_filter:
                filter_reasons.append("role_confusion")
            if qid in vague_qids_to_filter:
                filter_reasons.append("vague_communication_info")

            if filter_reasons:
                qid_filter_reasons[qid].extend(filter_reasons)
                # Store filter reason in sample metadata for reference
                if 'filter_reasons' not in sample:
                    sample['filter_reasons'] = filter_reasons
                dropped_samples.append(sample)
            else:
                passed_samples.append(sample)
        
        # Log do_nothing_keep tasks (similar to role confusion logging)
        for qid in sorted(do_nothing_keep_qids):
            mean_score = self._calculate_mean_score(question_groups[qid])
            logger.info(f"Kept {qid}: do_nothing task with success rate {mean_score:.3f} <= {DO_NOTHING_SUCCESS_RATE_THRESHOLD}")
        
        # Log summary of filtering reasons
        logger.info(f"TAU2 Bench filtering completed: {len(passed_samples)} passed, {len(dropped_samples)} dropped")
        logger.info(f"Filtering reasons summary:")
        for qid, reasons in sorted(qid_filter_reasons.items()):
            unique_reasons = list(set(reasons))
            logger.info(f"  {qid}: {', '.join(unique_reasons)}")
        
        # Print summary counts by reason
        reason_counts = defaultdict(int)
        for reasons in qid_filter_reasons.values():
            for reason in set(reasons):
                reason_counts[reason] += 1
        logger.info(f"Filtered qids by reason: {dict(reason_counts)}")
        logger.info(f"Kept do_nothing qids (success rate <= {DO_NOTHING_SUCCESS_RATE_THRESHOLD}): {len(do_nothing_keep_qids)}")
        
        return passed_samples, dropped_samples


    def _get_qids_solvable_by_do_nothing(self) -> List[str]:
        function_names_modifying_database = {
            'retail': ['cancel_pending_order', "exchange_delivered_order_items", "modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items"],
            'airline': ["book_reservation", "cancel_reservation", "send_certificate", "update_reservation_baggages", "update_reservation_flights", "update_reservation_passengers"],
        }

        result = []
        questions = Tau2BenchLoader().load_questions()

        for question in questions:
            qid = question.question_id
            
            domain = question.task_name
            if domain == 'telecom':
                continue
            actions = question.evaluation_criteria.get('actions', [])
            is_modifying_database = False
            if domain in ["retail", "airline"]:
                for action in actions:
                    if action['name'] in function_names_modifying_database[domain]:
                        is_modifying_database = True
                        break
                

            if len(actions) == 0 or not is_modifying_database:
                # Normalize to underscore format to match sample normalization
                normalized_qid = self._normalize_question_id(domain, qid)
                result.append(normalized_qid)

        return result

    def _get_qids_with_vague_communication_info(self) -> List[str]:
        result = []
        questions = Tau2BenchLoader().load_questions()

        for question in questions:
            qid = question.question_id
            
            domain = question.task_name
            communicate_info = question.evaluation_criteria.get('communicate_info', [])
            if communicate_info:
                for info in communicate_info:
                    if ' ' in info:
                        # Normalize to underscore format to match sample normalization
                        normalized_qid = self._normalize_question_id(domain, qid)
                        result.append(normalized_qid)
                        break

        return result

    def _group_samples_by_question(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Group samples by their question identifier."""
        question_groups = defaultdict(list)
        
        for sample in samples:
            qid = self._normalize_question_id(sample['task_name'], sample['meta']['id'])
            question_groups[qid].append(sample)
        
        return dict(question_groups)

    def _calculate_mean_score(self, question_samples: List[Dict]) -> float:
        """Calculate mean score for a question based on all model responses."""
        if not question_samples:
            return None
        
        # Extract scores for this question
        scores = []
        for sample in question_samples:
            # Try different possible score locations
            if 'eval_result' in sample and 'score' in sample['eval_result']:
                scores.append(sample['eval_result']['score'])
            elif 'eval_result' in sample and 'scores' in sample['eval_result']:
                scores.extend(sample['eval_result']['scores'])
            elif 'score' in sample:
                scores.append(sample['score'])
            elif 'scores' in sample:
                scores.extend(sample['scores'])
        
        if not scores:
            return None
        
        # Convert to numeric scores
        numeric_scores = []
        for score in scores:
            if isinstance(score, (int, float)):
                numeric_scores.append(float(score))
            elif isinstance(score, dict) and 'score' in score:
                try:
                    numeric_scores.append(float(score['score']))
                except (ValueError, TypeError):
                    continue
        
        if not numeric_scores:
            return None
        
        # Calculate mean score (success rate)
        return np.mean(numeric_scores)

    def _detect_role_confusion_from_user_message(self, content: str) -> bool:
        """Detect if user message shows role confusion by using agent-like language"""
        if not content or not isinstance(content, str):
            return False

        content_lower = content.lower()
        confusion_patterns = [
            r'\byour\s+(phone|device|computer|laptop|system|account|order|reservation|flight|booking|service|plan|bill|number)\b',
        ]

        for pattern in confusion_patterns:
            if re.search(pattern, content_lower):
                return True

        return False

    def _get_qids_with_role_confusion(self, samples: List[Dict], threshold: float) -> List[str]:
        """Get question IDs that have role confusion rate above threshold"""
        question_groups = self._group_samples_by_question(samples)
        qids_to_filter = []

        for qid, question_samples in question_groups.items():
            confusion_count = 0
            total_samples = len(question_samples)

            for sample in question_samples:
                has_confusion = False

                messages = []
                if 'conversation' in sample:
                    messages = sample['conversation']
                elif 'messages' in sample:
                    messages = sample['messages']
                elif 'response' in sample and isinstance(sample['response'], dict):
                    if 'conversation' in sample['response']:
                        messages = sample['response']['conversation']
                    elif 'messages' in sample['response']:
                        messages = sample['response']['messages']

                for message in messages:
                    if isinstance(message, dict) and message.get('role') == 'user' and 'content' in message:
                        content = message['content']
                        if self._detect_role_confusion_from_user_message(content):
                            has_confusion = True
                            break

                if has_confusion:
                    confusion_count += 1

            if total_samples > 0:
                confusion_rate = confusion_count / total_samples
                if confusion_rate > threshold:
                    qids_to_filter.append(qid)
                    logger.info(f"Filtering {qid}: role confusion rate {confusion_rate:.3f} > {threshold}")

        return qids_to_filter