import json
import os
import time

CHECKPOINT_FILE = "data/pipeline_checkpoint.json"


class CheckpointManager:
    def __init__(self, path=CHECKPOINT_FILE):
        self.path = path
        self.state = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {"phases": {}, "start_time": None}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2)

    def is_complete(self, phase_name):
        return self.state["phases"].get(phase_name, {}).get("status") == "complete"

    def mark_complete(self, phase_name, metadata=None):
        self.state["phases"][phase_name] = {
            "status": "complete",
            "completed_at": time.strftime("%H:%M:%S"),
            "metadata": metadata or {},
        }
        self.save()

    def mark_running(self, phase_name):
        self.state["phases"][phase_name] = {
            "status": "running",
            "started_at": time.strftime("%H:%M:%S"),
        }
        self.save()

    def get_next_phase(self, phase_order):
        for phase in phase_order:
            if not self.is_complete(phase):
                return phase
        return None

    def reset(self, phase_name=None):
        if phase_name:
            self.state["phases"].pop(phase_name, None)
        else:
            self.state = {"phases": {}, "start_time": time.strftime("%H:%M:%S")}
        self.save()

    def status_report(self, phase_order):
        for phase in phase_order:
            info = self.state["phases"].get(phase, {})
            status = info.get("status", "pending")
            if status == "complete":
                print(f"  [DONE] {phase} (completed at {info['completed_at']})")
            elif status == "running":
                print(f"  [WAS RUNNING] {phase} (started at {info.get('started_at', '?')})")
            else:
                print(f"  [PENDING] {phase}")
