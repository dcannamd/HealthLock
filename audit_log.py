# audit_log.py
# Logs every AI action to a local JSON file.
# Nothing is sent anywhere — this is a local audit trail only.

import json
from datetime import datetime
from pathlib import Path

LOG_FILE = "logs/audit_log.json"


def log_event(
    agent: str,
    file: str,
    action: str,
    result: str,
    confidence: str,
    flagged: bool,
):
    """Write a single audit log entry as a JSON line."""
    Path("logs").mkdir(exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "file": file,
        "action": action,
        "result": result[:200],  # truncate long results
        "confidence": confidence,
        "flagged_for_review": flagged,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
# audit_log.py
# Logs every AI action to a local JSON file.
# Nothing is sent anywhere — this is a local audit trail only.

import json
from datetime import datetime
from pathlib import Path

LOG_FILE = "logs/audit_log.json"


def log_event(
    agent: str,
    file: str,
    action: str,
    result: str,
    confidence: str,
    flagged: bool,
):
    """Write a single audit log entry as a JSON line."""
    Path("logs").mkdir(exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "file": file,
        "action": action,
        "result": result[:200],  # truncate long results
        "confidence": confidence,
        "flagged_for_review": flagged,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")