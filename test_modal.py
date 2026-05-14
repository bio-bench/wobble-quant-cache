"""Smoke test: run wobble-profile on Modal with a small model."""

import modal

app = modal.App("wobble-profile-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.40",
        "numpy>=1.24",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "datasets>=2.16",
        "accelerate>=0.26",
    )
    .add_local_dir("wobble", remote_path="/root/repo/wobble", copy=True)
    .add_local_dir("profiling", remote_path="/root/repo/profiling", copy=True)
    .env({"PYTHONPATH": "/root/repo"})
)


@app.function(image=image, gpu="T4", timeout=600)
def run_profile():
    import json
    import sys
    sys.path.insert(0, "/root/repo")

    from profiling.cli import main as profile_main

    sys.argv = [
        "wobble-profile",
        "--model", "Qwen/Qwen2.5-0.5B",
        "--output", "/root/output",
        "--n-texts", "20",
        "--dtype", "float16",
    ]

    try:
        profile_main()
    except SystemExit as e:
        if e.code and e.code != 0:
            raise

    from pathlib import Path
    output_dir = Path("/root/output")
    if output_dir.exists():
        for f in sorted(output_dir.iterdir()):
            print(f"  {f.name} ({f.stat().st_size:,} bytes)")
        report_path = output_dir / "profiling_report.json"
        if report_path.exists():
            print("\n=== REPORT ===")
            print(json.dumps(json.loads(report_path.read_text()), indent=2))


@app.local_entrypoint()
def main():
    run_profile.remote()
    print("\nDone.")
