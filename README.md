# QFF2026 UTSA grader

Auto-checks for the hands-on labs of **Qiskit Fall Fest 2026 @ UT San Antonio**.

Everything runs locally in the student's notebook: nothing is sent anywhere.

## Use it in a lab notebook

```python
%pip install -q --no-deps --force-reinstall git+https://github.com/vantnprof/QFF2026_UTSA_grader
from qff_utsa_grader.lab1 import check, scoreboard

check("A1")      # after each task
scoreboard()     # at the end
```

`check()` reads the student's variables (`coin`, `bell`, `theta_B1`, …) directly from the notebook, so the notebook only needs the task ID.

## Labs

| Module | Lab | Tasks |
|---|---|---|
| `qff_utsa_grader.lab1` | Lab 1 · Quantum coins & entangled pairs | A1–A3, B1–B3, C1–C6, bonus D1–D2 |
| `qff_utsa_grader.lab2` | Lab 2 · Find the target with Grover's algorithm | A1–A4, B1–B3, C1–C2, D1–D2, bonus E1–E2 |

## Fixing a check during the event

1. Edit the module, commit and push.
2. Students rerun the setup cell (`--force-reinstall` fetches the new version), then restart the kernel (**Kernel → Restart**) and run all cells, so Python loads the new code.

## Requirements

Python 3.10+ and Qiskit 2.x.
