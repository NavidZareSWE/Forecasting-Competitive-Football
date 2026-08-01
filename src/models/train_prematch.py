"""Pre-match half of the sweep: Model 1 (Task C) and Model 2 (Task R)."""

from run_models import (TASK_LABELS, run_classification_task,
                        run_regression_task, write_predictions, write_results)


def main():
    rows = []
    print(f"=== {TASK_LABELS['C']} ===")
    task_rows, predictions = run_classification_task("C")
    rows.extend(task_rows)
    write_predictions(predictions, "C")

    print(f"\n=== {TASK_LABELS['R']} ===")
    task_rows, predictions = run_regression_task("R")
    rows.extend(task_rows)
    write_predictions(predictions, "R")

    write_results(rows, "model_results_prematch.csv")


if __name__ == "__main__":
    main()
