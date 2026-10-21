# G1 Teleop Copilot

Right-hand orientation assistance for G1 teleoperation in Isaac Sim.

- Object selection and grasp-proposal processing.
- Operator-controlled orientation assistance and return to manual control.
- State checks, camera preview, and browser interface.

## Usage

Application code is in `teleop_assist/`. Entry points are `run_copilot.py` and `run_assist.py`.
Model dependencies are listed in `requirements-models.txt` and `requirements-pose.txt`.

```bash
python -m pytest -q tests/test_control_ownership.py
```

The documented setup uses manual hand positioning and fixed fingers.
