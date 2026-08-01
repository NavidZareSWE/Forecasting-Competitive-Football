"""In-play half of the sweep: Model 3, both labels, on the snapshot table."""

from run_models import (TASK_LABELS, run_classification_task,
                        run_regression_task, write_predictions, write_results)


def main():
    rows = []
    print(f"=== {TASK_LABELS['Lc']} ===")
    task_rows, predictions = run_classification_task("Lc")
    rows.extend(task_rows)
    write_predictions(predictions, "Lc")

    print(f"\n=== {TASK_LABELS['Lr']} ===")
    task_rows, predictions = run_regression_task("Lr")
    rows.extend(task_rows)
    write_predictions(predictions, "Lr")

    write_results(rows, "model_results_inplay.csv")


if __name__ == "__main__":
    main()
