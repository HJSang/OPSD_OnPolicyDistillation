"""Flyte batch workflow for the VTC-memory compression sweeps.

Runs a self-contained bash script on a single-GPU pod. The script sets up the
DeepSeek-OCR venv, then runs whatever sweep command is passed in. Results are
written to NFS at /shared/public/sharing/vtc_memory/results/ so they survive
after the pod exits.

Submit with (from the mldev_efficiency repo root):
    mldev run vtc_sweep -e <execution> -d <cluster> --crew-id 3330
"""
import logging
import os

from flytekit import Secret, task, workflow  # pyright: ignore[reportMissingImports]
from flytekitplugins.kfpytorch import PyTorch  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

SECRET_GROUP = "codefetcher-secret"
SECRET_NAME = "github_ae_proxy_url"
CONTAINER_IMAGE = "{{.image.mldev_verl_vllm_cu128_image}}"


def _make_executable(path: str) -> None:
    mode = os.stat(path).st_mode
    os.chmod(path, mode | 0o111)


@task(
    enable_nfs=True,
    secret_requests=[Secret(group=SECRET_GROUP, key=SECRET_NAME)],
    task_config=PyTorch(num_workers=0),
    instance_type="h100_2",
    container_image=CONTAINER_IMAGE,
    proxy_as="coreaiopt",
    enable_identity_certs=True,
)
def run_sweep(sweep_cmd: str, run_name: str) -> None:
    """Run a self-contained VTC-memory sweep on the pod.

    sweep_cmd: the python command(s) to execute inside the experiment dir,
               e.g. "python run_validation.py --dataset longmemeval ...".
    run_name:  used to name the log under the NFS results dir.
    """
    import subprocess

    # The experiment code ships as a resource under /home/jobuser/resources/.
    exp_dir = "/home/jobuser/resources/experiments/vtc_memory_validation"
    nfs = os.environ.get("VTC_NFS_ROOT", "/shared/public/sharing/vtc_memory")

    script = f"""#!/bin/bash
set -euo pipefail
cd {exp_dir}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# central config (VTC_* vars); allow the caller's env to override the defaults
export VTC_NFS_ROOT="${{VTC_NFS_ROOT:-{nfs}}}"
source ./env.sh

# data + model come from NFS staging
mkdir -p data
cp -n "$VTC_DATA_DIR"/*.json data/ || true

# DeepSeek-OCR runs from a prebuilt venv on NFS (transformers 4.46), invoked
# directly via its absolute path in the sweep command — no copy/pip needed
# (batch pods have no external internet).

echo "=== running sweep: {run_name} ==="
{sweep_cmd}

# persist results + logs to NFS
mkdir -p {nfs}/results
cp -f results_*.json {nfs}/results/ 2>/dev/null || true
cp -f dsocr_cache_*.json {nfs}/results/ 2>/dev/null || true
echo "=== sweep {run_name} done; results in {nfs}/results/ ==="
"""
    script_path = "/tmp/run_sweep.sh"
    with open(script_path, "w") as f:
        f.write(script)
    _make_executable(script_path)
    logger.info(f"Running sweep {run_name}")
    subprocess.run(script_path, shell=True, check=True)
    logger.info(f"Sweep {run_name} finished")


@workflow(namespace="training-coreai")
def vtc_sweep_workflow(sweep_cmd: str, run_name: str = "sweep") -> None:
    """Batch sweep workflow — see execution configs for concrete commands."""
    run_sweep(sweep_cmd=sweep_cmd, run_name=run_name)
