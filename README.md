# G1 Teleop Copilot

Operator-controlled orientation assistance for the G1's right hand in Isaac Sim. The application provides object selection, grasp-proposal processing, state checks, camera preview, and a browser interface. The documented setup keeps hand position manual and fingers fixed.


[Tool guide](docs/TOOLS.md): commands, inputs, outputs, and operating sequence.

## Initialize and test

Use Python 3.11:

```bash
git clone https://github.com/SpookyGeista/g1-teleop-copilot.git
cd g1-teleop-copilot
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-offline.txt
python -m pytest -q tests/test_control_ownership.py
cp configs/copilot.example.json configs/copilot.local.json
```

The offline test checks control ownership without starting a simulator.

## Configure the simulation stack

Edit `configs/copilot.local.json` and set these paths to existing installations:

| Field | Required location |
|---|---|
| `isaac_repository`, `isaac_python` | Unitree Isaac task checkout and its Python interpreter |
| `xr_repository`, `xr_python` | Unitree XR checkout and its Python interpreter |
| `grasp_repository`, `grasp_python` | GraspGenX checkout and its Python interpreter |
| `perception_python` | Python environment for the perception models |
| `descriptor`, `model_root` | Gripper descriptor and local model directory |

Install model-process packages in separate environments using `requirements-models.txt` and, for the pose backend, `requirements-pose.txt`. Use the existing compatible CUDA/PyTorch environment as their base. Provide the model checkpoints and gripper geometry listed in `ASSET_DEPENDENCIES.json` before starting assistance.

## Check the installation

With `isaac_python` configured:

```bash
python run_copilot.py --config configs/copilot.local.json --doctor
```

Resolve missing files or packages reported by the doctor first. For a configured stack, the isolated scripted simulation check is:

```bash
python run_copilot.py --config configs/copilot.local.json \
  --assist-sim --scripted-test --duration 30
```

This uses synthetic operator input. Physical-headset validation is separate. Runtime files belong in `runs/`; local configuration, TLS keys, and model weights should stay untracked.
