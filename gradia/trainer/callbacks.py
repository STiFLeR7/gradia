from typing import Dict, Any, List
import json
import time
import threading
import os
from pathlib import Path

# Shared lock for writing to the log file from multiple threads (Trainer vs SystemMonitor)
log_lock = threading.Lock()

class Callback:
    def on_train_begin(self, logs: Dict[str, Any] = {}): pass
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = {}): pass
    def on_train_end(self, logs: Dict[str, Any] = {}): pass

class EventLogger(Callback):
    """
    Logs events to a file which can be tailed by the UI server.
    Also keeps an in-memory buffer.
    """
    def __init__(self, log_dir: str):
        self.log_path = Path(log_dir) / "events.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear existing
        if self.log_path.exists():
            with log_lock:
                # Double check to avoid race if multiple loggers init (rare)
                if self.log_path.exists():
                    self.log_path.unlink()

    def _emit(self, event_type: str, data: Dict[str, Any]):
        payload = {
            "timestamp": time.time(),
            "type": event_type,
            "data": data
        }
        with log_lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(payload) + "\n")

    def on_train_begin(self, logs={}):
        self._emit("train_begin", logs)

    def on_epoch_end(self, epoch: int, logs={}):
        self._emit("epoch_end", {"epoch": epoch, **logs})

    def on_train_end(self, logs={}):
        self._emit("train_end", logs)
