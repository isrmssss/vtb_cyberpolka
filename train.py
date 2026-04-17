import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.checkpoint import CheckpointManager
from src.pipeline.phase7_optuna_tuning import run_optuna_tuning
from src.pipeline.phase8_postprocessing import run_postprocessing
from src.pipeline.phase9_inference import run_inference


class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def fmt(self, seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}min"
        else:
            return f"{seconds/3600:.1f}h"


PHASE_ORDER = [
    "phase7_optuna_tuning",
    "phase8_postprocessing",
    "phase9_inference",
]

PHASE_RUNNERS = {
    "phase7_optuna_tuning": run_optuna_tuning,
    "phase8_postprocessing": run_postprocessing,
    "phase9_inference": run_inference,
}


def main():
    timer = Timer()
    checkpoint = CheckpointManager()

    print("=" * 60)
    print("Day 3: CatBoost Optuna Tuning Pipeline")
    print(f"Total phases: {len(PHASE_ORDER)}")
    print(f"Started at: {time.strftime('%H:%M:%S')}")
    print("=" * 60)

    print("\nPipeline status:")
    checkpoint.status_report(PHASE_ORDER)

    while True:
        next_phase = checkpoint.get_next_phase(PHASE_ORDER)
        if next_phase is None:
            break

        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Running: {next_phase}")
        print(f"{'='*60}")

        checkpoint.mark_running(next_phase)

        try:
            result = PHASE_RUNNERS[next_phase]()
            checkpoint.mark_complete(next_phase, metadata=result)
            print(f"\n  {next_phase} completed in {timer.fmt(time.time() - t0)}")
        except Exception as e:
            print(f"\n  ERROR in {next_phase}: {e}")
            print(f"  Pipeline stopped. Run again to resume from here.")
            raise

    print("\n" + "=" * 60)
    print("Day 3 Complete!")

    tuning_phase = checkpoint.state["phases"].get("phase7_optuna_tuning", {})
    tuning_meta = tuning_phase.get("metadata", {})
    if tuning_meta:
        print(f"Macro ROC-AUC before: {tuning_meta.get('macro_auc_before', 'N/A'):.5f}")
        print(f"Macro ROC-AUC after: {tuning_meta.get('macro_auc_after', 'N/A'):.5f}")
        print(f"Improvement: {tuning_meta.get('improvement', 'N/A'):+.5f}")

    post_phase = checkpoint.state["phases"].get("phase8_postprocessing", {})
    post_meta = post_phase.get("metadata", {})
    if post_meta:
        print(f"Post-processed Macro ROC-AUC: {post_meta.get('macro_final', 'N/A'):.5f}")

    print(f"Total time: {timer.fmt(timer.elapsed())}")
    print(f"Finished at: {time.strftime('%H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
