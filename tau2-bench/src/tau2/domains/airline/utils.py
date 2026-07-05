from tau2.utils.utils import DATA_DIR, DOMAINS_DIR

AIRLINE_DATA_DIR = DOMAINS_DIR / "airline"
AIRLINE_DB_PATH = AIRLINE_DATA_DIR / "db.json"
AIRLINE_POLICY_PATH = AIRLINE_DATA_DIR / "policy.md"
AIRLINE_TASK_SET_PATH = AIRLINE_DATA_DIR / "tasks.json"

# Tau3-bench airline variant: vendored from sierra-research/tau3-bench.
# Pinned to data/tau2/domains_tau3/airline/ so it does not follow
# TAU2_DOMAINS_SUBDIR overrides.
AIRLINE_TAU3_DATA_DIR = DATA_DIR / "tau2" / "domains_tau3" / "airline"
AIRLINE_TAU3_DB_PATH = AIRLINE_TAU3_DATA_DIR / "db.json"
AIRLINE_TAU3_POLICY_PATH = AIRLINE_TAU3_DATA_DIR / "policy.md"
AIRLINE_TAU3_TASK_SET_PATH = AIRLINE_TAU3_DATA_DIR / "tasks.json"
