import os
import json
import time
from typing import Any, Dict


class Logger:
    def __init__(self, log_dir: str, tag: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{tag}.jsonl")
        self.tag = tag
        self.start = time.time()

    def log(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["_t"] = round(time.time() - self.start, 2)
        with open(self.path, "a") as f:
            f.write(json.dumps(payload) + "\n")

    def info(self, msg: str) -> None:
        print(f"[{self.tag}] {msg}", flush=True)
