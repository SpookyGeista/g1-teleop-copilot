# Copilot tools

Complete the [configuration steps](../README.md) first. Run commands from the repository root. The launcher reads `configs/copilot.local.json` to choose the simulator and model interpreters.

## Check dependencies

```bash
python run_copilot.py --config configs/copilot.local.json --doctor
```

Resolve the reported installation, geometry, and model-file problems before a simulation run.

## Scripted simulation check

```bash
python run_copilot.py --config configs/copilot.local.json \
  --assist-sim --scripted-test --duration 30
```

This uses synthetic operator input in isolated PhysX. It does not validate headset controls.

## Prepare a local viewer certificate

```bash
python tools/assist_tls.py --output runs/viewer_tls
```

This creates a local CA and server certificate/key. Keep them local and retain the CA for client trust configuration. The helper refuses to replace an existing output directory.

## Check and launch the XR session

```bash
python tools/copilot/launch_sim_xr.py \
  --check --config configs/copilot.local.json \
  --control-mode guarded_assist --live-ai --xr-camera \
  --camera-source right_wrist --runtime runs/operator_001 \
  --cert-file runs/viewer_tls/server.crt --key-file runs/viewer_tls/server.key \
  --duration 300
```

`--check` validates files and prints the launch plan without starting processes. Once it passes, replace `--check` with `--run` to start the session. Use a new runtime directory for each start. The default dashboard is port 8770; the XR input endpoint defaults to loopback on port 8012. For a headset on the LAN, explicitly choose `--host`, matching certificate names, and client trust.

## Operator workflow

Select the observed object in the desktop preview, choose a grasp, align/clutch the manual input, and confirm. Hold orientation assistance to use it; releasing it returns toward manual control. After target loss, select the object again before confirming another grasp. The current documented setup keeps hand position manual and fingers fixed.

## Saved-frame analysis

```bash
python tools/copilot/analyze_capture.py \
  --frame /path/to/captured_frame --output runs/capture_analysis \
  --prompt "red cylinder"
python run_copilot.py --config configs/copilot.local.json \
  --review runs/capture_analysis
```

The analysis needs compatible captured RGB-D/state data and the configured perception, grasp, and IK environments. Pass its interpreter/repository overrides when using different local paths. `run_assist.py` is the separate assisted-simulation entry point; inspect its options with `python run_assist.py --help` in the configured Isaac environment.
