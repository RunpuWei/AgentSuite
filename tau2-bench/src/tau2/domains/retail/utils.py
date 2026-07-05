from tau2.utils.utils import DATA_DIR, DOMAINS_DIR

RETAIL_DATA_DIR = DOMAINS_DIR / "retail"
RETAIL_DB_PATH = RETAIL_DATA_DIR / "db.json"
RETAIL_POLICY_PATH = RETAIL_DATA_DIR / "policy.md"
RETAIL_TASK_SET_PATH = RETAIL_DATA_DIR / "tasks.json"

# Tau3-bench retail variant: vendored from sierra-research/tau3-bench.
# Pinned to data/tau2/domains_tau3/retail/ so it does not follow
# TAU2_DOMAINS_SUBDIR overrides.
RETAIL_TAU3_DATA_DIR = DATA_DIR / "tau2" / "domains_tau3" / "retail"
RETAIL_TAU3_DB_PATH = RETAIL_TAU3_DATA_DIR / "db.json"
RETAIL_TAU3_POLICY_PATH = RETAIL_TAU3_DATA_DIR / "policy.md"
RETAIL_TAU3_TASK_SET_PATH = RETAIL_TAU3_DATA_DIR / "tasks.json"
