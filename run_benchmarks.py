#!/usr/bin/env python3
"""
Universal Benchmark Runner for AgentSuite
A one-click solution to run all LLM tool-use benchmarks
"""

import os
import sys
import json
import argparse
import subprocess
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple, List
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Temperature policy — single source of truth (issue #1).
#
# The entry point decides the effective temperature for each model and pushes it
# down to the benchmarks via environment variables. Downstream benchmark code must
# read AGENTSUITE_TEMPERATURE / AGENTSUITE_USER_TEMPERATURE and execute
# it verbatim — no per-benchmark "if thinking -> 1.0" overrides. If a model cannot
# use the supplied temperature, it should fail loudly rather than be silently patched.
# ---------------------------------------------------------------------------
def requires_default_temperature(model_name: str) -> bool:
    """True if a model only accepts the provider's default temperature
    (reasoning / extended-thinking models)."""
    m = (model_name or "").lower()
    # OpenAI reasoning families: gpt-5 / gpt-5-mini / gpt-5-nano / gpt-5.2 (not gpt-55)
    if re.search(r"gpt-5(?!\d)", m):
        return True
    # OpenAI o-series reasoning: o1 / o3 / o4 (not o5, not the 'o' in gpt-4o)
    if re.search(r"(?:^|/)o[1-4](?:-|$)", m):
        return True
    # Anthropic / Gemini extended thinking
    if "thinking-on" in m:
        return True
    return False


def normalize_temperature(model_name: str, temperature: float) -> float:
    """Effective temperature for a model: reasoning/thinking models must use the
    provider default (1.0); everything else uses the requested value."""
    return 1.0 if requires_default_temperature(model_name) else temperature


class BenchmarkRunner:
    """Main benchmark runner class"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        output_dir: Optional[str] = None,
        temperature: float = 0.0,
        proc_num: int = 4,
        user_model: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.proc_num = proc_num
        self.user_model = user_model or "openai/gpt-4o-20240806"  # Default user model
        self.limit = limit
        self.cleaned = False

        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir
            else Path(f"results_{timestamp}").resolve()
        )
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Store root directory
        self.root_dir = Path.cwd()

        # Track results
        self.results = {}

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "benchmark_run.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def get_provider_from_model(self) -> str:
        """Determine provider from model name"""
        return self.get_provider_from_model_name(self.model_name)

    def get_provider_from_model_name(self, model_name: str) -> str:
        """Determine provider from any model name"""
        model_lower = model_name.lower()
        if "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "mistral" in model_lower:
            return "mistral"
        elif "together" in model_lower:
            return "together_ai"
        else:
            return "openai"

    def setup_environment(self, benchmark_dir: Path):
        """Setup environment variables for benchmark execution"""
        env = os.environ.copy()

        # Unified API key pattern - all benchmarks use OpenAI-compatible API_KEY/BASE_URL
        env_vars = {
            "API_KEY": self.api_key,  # Primary agent model (OpenAI-compatible)
            "BASE_URL": self.base_url,  # Primary agent model base URL
            "USER_API_KEY": os.getenv("USER_API_KEY", self.api_key),  # User model (defaults to same as agent)
            "USER_BASE_URL": os.getenv("USER_BASE_URL", self.base_url),  # User model base URL
        }

        # Add RAPID_API_KEY if it exists in environment
        rapid_api_key = os.getenv("RAPID_API_KEY")
        if rapid_api_key:
            env_vars["RAPID_API_KEY"] = rapid_api_key

        # Temperature policy decided here (single source of truth); benchmarks read
        # these and execute verbatim. See requires_default_temperature() above.
        env_vars["AGENTSUITE_TEMPERATURE"] = str(
            normalize_temperature(self.model_name, self.temperature)
        )
        env_vars["AGENTSUITE_USER_TEMPERATURE"] = str(
            normalize_temperature(self.user_model, self.temperature)
        )

        # Cleaned mode (--cleaned): switch to fixed data directories
        if self.cleaned:
            env_vars["BFCL_DATA_DIR"] = "data_fixed"
            env_vars["TAU2_DOMAINS_SUBDIR"] = "domains_fixed"
            env_vars["TAU_BENCH_USE_FIXED"] = "1"

        env.update(env_vars)
        return env

    def create_env_file(self, benchmark_dir: Path, custom_vars: Optional[Dict] = None):
        """Create .env file for benchmarks that need it"""
        env_file = benchmark_dir / ".env"

        default_vars = {
            "API_KEY": self.api_key,  # Primary agent model (OpenAI-compatible)
            "BASE_URL": self.base_url,  # Primary agent model base URL
            "USER_API_KEY": os.getenv("USER_API_KEY", self.api_key),  # User model (OpenAI-compatible)
            "USER_BASE_URL": os.getenv("USER_BASE_URL", self.base_url),  # User model base URL
        }

        if custom_vars:
            default_vars.update(custom_vars)

        with open(env_file, "w") as f:
            for key, value in default_vars.items():
                f.write(f"{key}={value}\n")

        self.logger.info(f"Created .env file in {benchmark_dir}")

    def run_command(
        self, command: str, cwd: Path, timeout: int = 3600
    ) -> Tuple[bool, str]:
        """Execute a command with real-time output and return success status and output"""
        try:
            self.logger.info(f"Executing: {command}")
            self.logger.info(f"Working directory: {cwd}")

            env = self.setup_environment(cwd)

            # Use Popen for real-time output
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

            output_lines = []

            # Read output line by line in real-time
            try:
                while True:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        break
                    if output:
                        print(output.rstrip())  # Print to terminal in real-time
                        output_lines.append(output)

                # Wait for process to complete
                return_code = process.wait(timeout=timeout)
                full_output = "".join(output_lines)

                if return_code == 0:
                    self.logger.info(f"Command completed successfully")
                    return True, full_output
                else:
                    self.logger.error(f"Command failed with return code {return_code}")
                    return False, full_output

            except subprocess.TimeoutExpired:
                process.kill()
                self.logger.error(f"Command timed out after {timeout} seconds")
                return False, f"Command timed out after {timeout} seconds"

        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return False, f"Error: {str(e)}"

    def run_drafterbench(self) -> bool:
        """Run DrafterBench"""
        benchmark_name = "DrafterBench"
        benchmark_dir = self.root_dir / "DrafterBench"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        self.logger.info(f"Running {benchmark_name}...")

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        provider = self.get_provider_from_model()
        command = f"python evaluation.py --model {self.model_name} --model-provider {provider} --temperature {self.temperature} --vllm_url {self.base_url} --proc_num {self.proc_num} --result_dir {benchmark_output_dir}"
        if self.limit:
            command += f" --limit {self.limit}"

        success, output = self.run_command(command, benchmark_dir)

        # Save output
        output_file = benchmark_output_dir / "output.log"
        with open(output_file, "w") as f:
            f.write(output)
        return success

    def run_nexusbench(self) -> bool:
        """Run NexusBench"""
        benchmark_name = "NexusBench"
        benchmark_dir = self.root_dir / "NexusBench"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        self.logger.info(f"Running {benchmark_name}...")

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        # Create .env file
        self.create_env_file(benchmark_dir)

        command = f"nexusbench --client OpenAI --base_url {self.base_url} --api_key {self.api_key} --model {self.model_name} --benchmarks all --num_samples_parallel {self.proc_num} --output-dir {benchmark_output_dir}"
        if self.limit:
            command += f" --limit {self.limit}"

        success, output = self.run_command(command, benchmark_dir)

        # Save output
        output_file = benchmark_output_dir / "output.log"
        with open(output_file, "w") as f:
            f.write(output)

        # NexusBench now saves results locally via --output-dir parameter
        return success

    def run_cfbench(self) -> bool:
        """Run CFBench"""
        benchmark_name = "CFBench"
        benchmark_dir = self.root_dir / "ComplexFuncBench"

        if not benchmark_dir.exists():
            self.logger.warning(
                f"{benchmark_name} source directory not found, skipping..."
            )
            return False

        self.logger.info(f"Running {benchmark_name}...")

        # Create .env file for ComplexFuncBench
        self.create_env_file(benchmark_dir)

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        command = f"python evaluation.py --model_name={self.model_name} --proc_num {self.proc_num} --output-dir {benchmark_output_dir} --eval_model {self.user_model}"
        if self.limit:
            command += f" --limit {self.limit}"
        if self.cleaned:
            # Fixed split: run on the repaired dataset (manifest FIX rows applied)
            # and drop the REMOVE-only set (manifest-derived, unfixable tasks).
            command += " --input_file data_fixed/ComplexFuncBench.jsonl"
            command += f" --filter-ids {benchmark_dir / 'cfbench_remove_ids.csv'}"

        success, output = self.run_command(command, benchmark_dir)

        # Save output
        output_file = benchmark_output_dir / "output.log"
        with open(output_file, "w") as f:
            f.write(output)

        # Run print_results.py to compute and display metrics
        result_jsonl = benchmark_output_dir / "full-1000.jsonl"
        if result_jsonl.exists():
            self.logger.info("Computing CFBench metrics...")
            eval_command = f"python print_results.py --result_dir {result_jsonl}"
            eval_success, eval_output = self.run_command(eval_command, benchmark_dir)
            eval_output_file = benchmark_output_dir / "metrics.log"
            with open(eval_output_file, "w") as f:
                f.write(eval_output)

        return success

    def run_multichallenge(self) -> bool:
        """Run MultiChallenge"""
        benchmark_name = "MultiChallenge"
        benchmark_dir = self.root_dir / "multi_challenge"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        self.logger.info(f"Running {benchmark_name}...")

        # Create .env file for MultiChallenge
        self.create_env_file(benchmark_dir)

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        provider = "openai"

        # Extract model name without provider prefix
        model_safe_name = self.model_name.replace("/", "_")

        command = f"python main.py --model-provider {provider} --provider-args model={self.model_name} temp={normalize_temperature(self.model_name, self.temperature)} --attempts 1 --output-file {benchmark_output_dir}/{model_safe_name}_evaluation_results.txt --raw {benchmark_output_dir}/{model_safe_name}_detailed_results.csv --max-workers_response_gen {self.proc_num} --max-workers_eval {self.proc_num} --eval-model {self.user_model}"
        if self.limit:
            command += f" --limit {self.limit}"

        success, output = self.run_command(command, benchmark_dir)

        # Save output
        output_file = benchmark_output_dir / "output.log"
        with open(output_file, "w") as f:
            f.write(output)
        return success

    def run_acebench(self) -> bool:
        """Run ACEBench"""
        benchmark_name = "ACEBench"
        benchmark_dir = self.root_dir / "ACEBench"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        # Create benchmark-specific output directory if not exists
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Running {benchmark_name}...")

        # ACEBench uses GPT_API_KEY/GPT_BASE_URL and GPT_AGENT_API_KEY internally
        user_api_key = os.getenv("USER_API_KEY", self.api_key)
        user_base_url = os.getenv("USER_BASE_URL", self.base_url)
        self.create_env_file(benchmark_dir, custom_vars={
            "GPT_API_KEY": user_api_key,
            "GPT_BASE_URL": user_base_url,
            "GPT_AGENT_API_KEY": self.api_key,
            "GPT_AGENT_BASE_URL": self.base_url,
        })

        command = f"python generate.py --model {self.model_name} --user-model {self.user_model} --temperature {self.temperature} --category normal special agent --language en --num-threads {self.proc_num} --output-dir {benchmark_output_dir}"
        if self.limit:
            command += f" --limit {self.limit}"
        if self.cleaned:
            # Fixed split drops the REMOVE-only set (manifest-derived).
            command += f" --filter-ids {benchmark_dir / 'acebench_remove_ids.csv'}"

        success, output = self.run_command(command, benchmark_dir)

        # Save generation output
        output_file = benchmark_output_dir / "generate_output.log"
        with open(output_file, "w") as f:
            f.write(output)

        # Run evaluation if generation succeeded
        if success:
            self.logger.info("Running ACEBench evaluation...")
            eval_command = f"python eval_main.py --model {self.model_name} --category normal special agent --language en --output-dir {benchmark_output_dir}"
            if self.cleaned:
                eval_command += " --cleaned"

            eval_success, eval_output = self.run_command(eval_command, benchmark_dir)

            # Save evaluation output
            eval_output_file = benchmark_output_dir / "eval_output.log"
            with open(eval_output_file, "w") as f:
                f.write(eval_output)

        return success

    def run_taubench(self) -> bool:
        """Run TauBench"""
        benchmark_name = "TauBench"
        benchmark_dir = self.root_dir / "tau-bench"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        self.logger.info(f"Running {benchmark_name}...")

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        # tau-bench uses OPENAI_API_KEY/OPENAI_API_BASE internally
        self.create_env_file(benchmark_dir, custom_vars={
            "OPENAI_API_KEY": self.api_key,
            "OPENAI_API_BASE": self.base_url,
        })

        provider = self.get_provider_from_model()
        user_provider = self.get_provider_from_model_name(self.user_model)

        # Load REMOVE ids for the Fixed split (manifest-derived; tau-bench has none)
        tau_filter_ids = {}
        if self.cleaned:
            import csv
            with open(benchmark_dir / "taubench_remove_ids.csv") as f:
                for row in csv.DictReader(f):
                    tid = row["task_id"]
                    parts = tid.split("_", 1)
                    tau_filter_ids.setdefault(parts[0], set()).add(int(parts[1]))

        # Run both retail and airline environments
        all_success = True
        all_outputs = []

        for env_type in ["retail", "airline"]:
            self.logger.info(f"Running TauBench on {env_type} environment...")
            command = f"python run.py --agent-strategy tool-calling --env {env_type} --model {self.model_name} --model-provider {provider} --user-model {self.user_model} --user-model-provider {user_provider} --user-strategy llm --max-concurrency {self.proc_num} --temperature {self.temperature} --base-url {self.base_url} --log-dir {benchmark_output_dir}"
            if self.limit:
                command += f" --end-index {self.limit}"
            if self.cleaned and env_type in tau_filter_ids:
                # tau-bench --task-ids takes integer indices; build include list
                # Total tasks: airline=50, retail=115
                total = 50 if env_type == "airline" else 115
                exclude = tau_filter_ids[env_type]
                include = [i for i in range(total) if i not in exclude]
                command += " --task-ids " + " ".join(str(i) for i in include)

            env_success, env_output = self.run_command(
                command, benchmark_dir, timeout=7200
            )
            all_success = all_success and env_success
            all_outputs.append(f"=== {env_type.upper()} ENVIRONMENT ===\n{env_output}")

            if not env_success:
                self.logger.error(f"TauBench failed on {env_type} environment")

        # Combine all outputs
        output = "\n\n".join(all_outputs)
        success = all_success

        # Save output
        output_file = benchmark_output_dir / "output.log"
        with open(output_file, "w") as f:
            f.write(output)

        return success

    def _run_bfcl_with_category(self, benchmark_name: str, test_category: str, fc: bool = False) -> bool:
        """Run BFCL with specific test category.

        Args:
            fc: If True, append '-FC' to the model name so the dynamic registration
                creates a function-calling config (is_fc_model=True, underscore_to_dot=True).
        """
        benchmark_dir = self.root_dir / "gorilla" / "berkeley-function-call-leaderboard"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        # For FC mode, append '-FC' suffix — BFCL's dynamic registration detects this
        bfcl_model_name = f"{self.model_name}-FC" if fc else self.model_name

        self.logger.info(f"Running {benchmark_name} (category: {test_category}, mode: {'FC' if fc else 'Prompt'})...")

        # Create .env file for BFCL
        self.create_env_file(benchmark_dir)

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        # Run generation
        command = f"bfcl generate --model {bfcl_model_name} --test-category {test_category} --num-threads {self.proc_num} --temperature {self.temperature} --result-dir {benchmark_output_dir}/result"
        if self.limit:
            command += f" --limit {self.limit}"
        if self.cleaned:
            # Fixed split drops the REMOVE-only set, selected per BFCL version
            # (benchmark_name is BFCL_v3 / BFCL_v3_FC / BFCL_v4 / BFCL_v4_FC).
            bfcl_version = "bfcl_v4" if "v4" in benchmark_name.lower() else "bfcl_v3"
            command += f" --filter-ids {benchmark_dir / f'{bfcl_version}_remove_ids.csv'}"

        success, output = self.run_command(
            command, benchmark_dir, timeout=7200
        )

        # Save generation output
        output_file = benchmark_output_dir / "generate_output.log"
        with open(output_file, "w") as f:
            f.write(output)

        # Run evaluation if generation succeeded
        if success:
            self.logger.info(f"Running {benchmark_name} evaluation...")
            eval_command = f"bfcl evaluate --model {bfcl_model_name} --result-dir {benchmark_output_dir}/result --score-dir {benchmark_output_dir}/score"
            if self.limit:
                eval_command += " --partial-eval"

            eval_success, eval_output = self.run_command(eval_command, benchmark_dir)

            # Save evaluation output
            eval_output_file = benchmark_output_dir / "eval_output.log"
            with open(eval_output_file, "w") as f:
                f.write(eval_output)

        return success

    def run_bfcl_v3(self) -> bool:
        """Run BFCL-v3 (single_turn + multi_turn) in prompting mode"""
        return self._run_bfcl_with_category("BFCL_v3", "single_turn,multi_turn")

    def run_bfcl_v3_fc(self) -> bool:
        """Run BFCL-v3 (single_turn + multi_turn) in function-calling mode"""
        return self._run_bfcl_with_category("BFCL_v3_FC", "single_turn,multi_turn", fc=True)

    def run_bfcl_v4(self) -> bool:
        """Run BFCL-v4 (agentic) in prompting mode"""
        return self._run_bfcl_with_category("BFCL_v4", "agentic")

    def run_bfcl_v4_fc(self) -> bool:
        """Run BFCL-v4 (agentic) in function-calling mode"""
        return self._run_bfcl_with_category("BFCL_v4_FC", "agentic", fc=True)

    def run_tau2bench(self) -> bool:
        """Run Tau2Bench (tau2-bench)"""
        benchmark_name = "Tau2Bench"
        benchmark_dir = self.root_dir / "tau2-bench"

        if not benchmark_dir.exists():
            self.logger.warning(f"{benchmark_name} directory not found, skipping...")
            return False

        self.logger.info(f"Running {benchmark_name}...")

        # Create benchmark-specific output directory
        benchmark_output_dir = self.output_dir / benchmark_name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        # tau2-bench uses OPENAI_API_KEY/OPENAI_API_BASE internally
        self.create_env_file(benchmark_dir, custom_vars={
            "OPENAI_API_KEY": self.api_key,
            "OPENAI_API_BASE": self.base_url,
        })

        # Load REMOVE ids for the Fixed split (manifest-derived)
        tau2_filter_ids = {}
        if self.cleaned:
            import csv
            with open(benchmark_dir / "tau2bench_remove_ids.csv") as f:
                for row in csv.DictReader(f):
                    tid = row["task_id"]
                    # Split "airline_14" -> domain="airline", id="14"
                    # But telecom IDs are complex like "telecom_[...]", split on first _
                    parts = tid.split("_", 1)
                    domain_key = parts[0]
                    task_id = parts[1] if len(parts) > 1 else tid
                    tau2_filter_ids.setdefault(domain_key, set()).add(task_id)

        # Run all three domains: retail, airline, telecom
        all_success = True
        all_outputs = []

        domains = ["retail", "airline", "telecom"]

        for domain in domains:
            self.logger.info(f"Running Tau2Bench on {domain} domain...")
            agent_temp = normalize_temperature(self.model_name, self.temperature)
            user_temp = normalize_temperature(self.user_model, self.temperature)
            command = (
                f"tau2 run --domain {domain} "
                f"--agent-llm {self.model_name} --agent-llm-args '{{\"temperature\": {agent_temp}}}' "
                f"--user-llm {self.user_model} --user-llm-args '{{\"temperature\": {user_temp}}}' "
                f"--num-trials 1 --max-concurrency {self.proc_num} --base-url {self.base_url} "
                f"--save-to {benchmark_output_dir}/{domain}_results"
            )
            if self.limit:
                command += f" --num-tasks {self.limit}"
            if self.cleaned and domain in tau2_filter_ids:
                # Build include list: load all task IDs from fixed data, exclude filtered ones
                import json
                tasks_file = benchmark_dir / "data" / "tau2" / "domains_fixed" / domain / "tasks.json"
                with open(tasks_file) as f:
                    all_tasks = json.load(f)
                all_ids = [t["id"] for t in all_tasks]
                include_ids = [tid for tid in all_ids if tid not in tau2_filter_ids[domain]]
                if include_ids:
                    command += " --task-ids " + " ".join(include_ids)

            domain_success, domain_output = self.run_command(
                command, benchmark_dir, timeout=7200
            )
            all_success = all_success and domain_success
            all_outputs.append(f"=== {domain.upper()} DOMAIN ===\n{domain_output}")

            if not domain_success:
                self.logger.error(f"Tau2Bench failed on {domain} domain")

        # Combine all outputs
        output = "\n\n".join(all_outputs)

        # Save output
        output_file = benchmark_output_dir / "output.log"
        with open(output_file, "w") as f:
            f.write(output)

        return all_success

    def run_single_benchmark(self, benchmark_name: str) -> bool:
        """Run a single benchmark"""
        benchmark_methods = {
            "drafterbench": self.run_drafterbench,
            "nexusbench": self.run_nexusbench,
            "cfbench": self.run_cfbench,
            "multichallenge": self.run_multichallenge,
            "acebench": self.run_acebench,
            "taubench": self.run_taubench,
            "tau2bench": self.run_tau2bench,
            "bfcl_v3": self.run_bfcl_v3,
            "bfcl_v3_fc": self.run_bfcl_v3_fc,
            "bfcl_v4": self.run_bfcl_v4,
            "bfcl_v4_fc": self.run_bfcl_v4_fc,
        }

        if benchmark_name not in benchmark_methods:
            self.logger.error(f"Unknown benchmark: {benchmark_name}")
            return False

        # --cleaned not yet implemented for these benchmarks
        if self.cleaned and benchmark_name in ("drafterbench", "nexusbench", "multichallenge"):
            self.logger.error(f"--cleaned is not yet implemented for {benchmark_name}")
            return False

        try:
            return benchmark_methods[benchmark_name]()
        except Exception as e:
            self.logger.error(f"Error running {benchmark_name}: {e}")
            return False

    def run_all_benchmarks(self, concurrent: bool = False) -> Dict[str, bool]:
        """Run all benchmarks"""
        benchmarks = [
            "drafterbench",
            "bfcl_v3",
            "bfcl_v4",
            "nexusbench",
            "cfbench",
            "acebench",
            "multichallenge",
            "taubench",
            "tau2bench",
        ]

        results = {}
        start_time = time.time()

        if concurrent:
            # Run benchmarks concurrently (be careful with API limits)
            self.logger.info("Running benchmarks concurrently...")

            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_benchmark = {
                    executor.submit(self.run_single_benchmark, benchmark): benchmark
                    for benchmark in benchmarks
                }

                for future in as_completed(future_to_benchmark):
                    benchmark = future_to_benchmark[future]
                    try:
                        result = future.result()
                        results[benchmark] = result
                        status = "✓" if result else "✗"
                        self.logger.info(f"Completed: {benchmark} {status}")

                    except Exception as e:
                        self.logger.error(f"{benchmark} failed with exception: {e}")
                        results[benchmark] = False

        else:
            # Run benchmarks sequentially
            self.logger.info("Running benchmarks sequentially...")
            for i, benchmark in enumerate(benchmarks, 1):
                self.logger.info(
                    f"Starting benchmark {i}/{len(benchmarks)}: {benchmark}"
                )

                results[benchmark] = self.run_single_benchmark(benchmark)
                status = "✓" if results[benchmark] else "✗"
                self.logger.info(f"Completed {benchmark}: {status}")

        # Final summary
        total_duration = time.time() - start_time
        successful = sum(results.values())
        self.logger.info(
            f"All benchmarks completed: {successful}/{len(benchmarks)} successful (took {total_duration:.0f}s)"
        )
        return results

    def generate_summary(self, results: Dict[str, bool]):
        """Generate execution summary"""
        summary_file = self.output_dir / "summary.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "base_url": self.base_url,
            "results": results,
            "total_benchmarks": len(results),
            "successful_benchmarks": sum(results.values()),
            "failed_benchmarks": len(results) - sum(results.values()),
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Also create human-readable summary
        text_summary_file = self.output_dir / "summary.txt"
        with open(text_summary_file, "w") as f:
            f.write("Benchmark Execution Summary\n")
            f.write("==========================\n\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write(f"Model: {summary['model']}\n")
            f.write(f"Base URL: {summary['base_url']}\n")
            f.write(f"Total Benchmarks: {summary['total_benchmarks']}\n")
            f.write(f"Successful: {summary['successful_benchmarks']}\n")
            f.write(f"Failed: {summary['failed_benchmarks']}\n\n")

            f.write("Individual Results:\n")
            f.write("------------------\n")
            for benchmark, success in results.items():
                status = "✓ PASSED" if success else "✗ FAILED"
                f.write(f"{benchmark:15} {status}\n")

        self.logger.info(f"Summary saved to {summary_file} and {text_summary_file}")


def load_env_config():
    """Load configuration from .env file if it exists"""
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file, override=True)
        print(f"Loaded configuration from {env_file}")
    else:
        print(
            "No .env file found. Please copy .env.example to .env and configure API credentials."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Universal Benchmark Runner for AgentSuite"
    )
    parser.add_argument(
        "model_name", nargs="?", default=None, help="Name of the model (e.g., openai/gpt-4o-20240806)"
    )
    parser.add_argument(
        "--benchmark", help="Specific benchmark to run (default: all)", default="all"
    )
    parser.add_argument(
        "--output-dir", default="./results", help="Output directory for results"
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Run benchmarks concurrently (use with caution for API limits)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model generation (default: 0.0)",
    )
    parser.add_argument(
        "--proc-num",
        type=int,
        default=4,
        help="Number of processes/threads for parallel execution (default: 4)",
    )
    parser.add_argument(
        "--user-model",
        help="User model for benchmarks that need it (default: from .env USER_MODEL or openai/gpt-4o-20240806)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per benchmark for quick validation",
    )
    parser.add_argument(
        "--fixed",
        "--cleaned",  # deprecated alias, kept for backward compatibility
        dest="fixed",
        action="store_true",
        help="Fixed split: use fixed task sets/data with known-problematic tasks excluded",
    )
    parser.add_argument(
        "--list-benchmarks", action="store_true", help="List available benchmarks"
    )

    # Load .env configuration first
    load_env_config()

    args = parser.parse_args()

    if args.list_benchmarks:
        print("Available benchmarks:")
        benchmarks = [
            "drafterbench - DrafterBench technical drawing revision tasks",
            "nexusbench - NexusBench function calling benchmark",
            "cfbench - CFBench comprehensive function calling evaluation",
            "multichallenge - MultiChallenge conversation evaluation",
            "acebench - ACEBench comprehensive agent evaluation",
            "taubench - TauBench tool-agent-user interaction (retail + airline)",
            "tau2bench - Tau2Bench conversational agents (retail + airline + telecom)",
            "bfcl_v3 - BFCL-v3 single-turn + multi-turn (prompting mode)",
            "bfcl_v3_fc - BFCL-v3 single-turn + multi-turn (function-calling mode)",
            "bfcl_v4 - BFCL-v4 agentic (prompting mode)",
            "bfcl_v4_fc - BFCL-v4 agentic (function-calling mode)",
        ]
        for benchmark in benchmarks:
            print(f"  {benchmark}")
        return

    if not args.model_name:
        parser.error("model_name is required when running benchmarks")

    # Get API credentials from .env file (required)
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        print("Error: API_KEY not found in .env file. Please set API_KEY in .env file")
        sys.exit(1)

    if not base_url:
        print(
            "Error: BASE_URL not found in .env file. Please set BASE_URL in .env file"
        )
        sys.exit(1)

    # Get user model from command line or .env file
    user_model = args.user_model or os.getenv("USER_MODEL")

    os.makedirs(args.output_dir, exist_ok=True)
    runner = BenchmarkRunner(
        api_key=api_key,
        base_url=base_url,
        model_name=args.model_name,
        output_dir=args.output_dir,
        temperature=args.temperature,
        proc_num=args.proc_num,
        user_model=user_model,
        limit=args.limit,
    )
    runner.cleaned = args.fixed

    runner.logger.info("Starting benchmark execution...")
    runner.logger.info(f"Model: {args.model_name}")
    runner.logger.info(f"User Model: {runner.user_model}")
    runner.logger.info(f"Base URL: {base_url}")
    runner.logger.info(f"Temperature: {args.temperature}")
    runner.logger.info(f"Processes/Threads: {args.proc_num}")
    runner.logger.info(f"Output directory: {runner.output_dir}")
    if args.limit:
        runner.logger.info(f"Sample limit: {args.limit}")
    if args.fixed:
        runner.logger.info("Fixed mode (--fixed): using fixed task sets/data")

    try:
        if args.benchmark == "all":
            results = runner.run_all_benchmarks(concurrent=args.concurrent)
        else:
            benchmark_result = runner.run_single_benchmark(args.benchmark)
            results = {args.benchmark: benchmark_result}

        # Generate summary
        runner.generate_summary(results)

        # Print results
        print(f"\n{'=' * 50}")
        print("BENCHMARK EXECUTION COMPLETED")
        print(f"{'=' * 50}")
        print(f"Results saved in: {runner.output_dir}")

        successful = sum(results.values())
        total = len(results)

        print(f"\nOverall Results: {successful}/{total} benchmarks passed")

        for benchmark, success in results.items():
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"  {benchmark:15} {status}")

        if successful < total:
            print(
                f"\n  {total - successful} benchmark(s) failed. Check logs in {runner.output_dir}"
            )
            sys.exit(1)
        else:
            print(f"\n All benchmarks completed successfully!")

    except KeyboardInterrupt:
        runner.logger.info("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        runner.logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
