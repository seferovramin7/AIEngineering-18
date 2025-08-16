# Homework: Transfer Learning & Data Augmentation (1 task)

Follow the steps to complete the assignment.

## Prerequisites

- Python 3.9+
- TensorFlow (install per `README_APP.md`)

# Task 1: Fine-tune and report

1) Run the training demo for 2 epochs:

```
python app/main.py demo-train --epochs 2 --image-size 160
```

2) After training finishes, note and record:

- Final training accuracy
- Final validation accuracy
- A brief sentence on whether validation improved during fine-tuning

3) Modify the fine-tuning depth in the app by changing how many layers remain trainable (e.g., last 10 layers, last 40 layers). Re-run for 2 epochs and compare results. Hints:
   - Look for the loop that freezes layers: `for layer in base_model.layers[:-20]: ...`
4) Submit a short report (200–300 words) including:

- What fine-tuning depth you tried
- The accuracies you observed
- Your interpretation: which setting worked best and why you think that happened
- One idea for a different augmentation you would try next time

## Deliverables

- A text or PDF report titled `transfer_learning_report.*`
- Optional: your modified `app/main.py` if you made changes

## Grading (simple rubric)

- Runs successfully: 30%
- Report completeness and clarity: 40%
- Quality of analysis and reasoning: 30%
