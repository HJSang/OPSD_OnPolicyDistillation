#!/usr/bin/env python3
import argparse
import getpass
import importlib
import importlib.metadata
import platform

import torch


EXPECTED = {
    "accelerate": "1.12.0",
    "cachetools": "6.2.6",
    "cbor2": "6.1.3",
    "datasets": "4.5.0",
    "llvmlite": "0.44.0",
    "numba": "0.61.2",
    "nvidia-cudnn-cu12": "9.16.0.29",
    "nvidia-nccl-cu12": "2.27.5",
    "pillow": "11.3.0",
    "qwen-vl-utils": "0.0.14",
    "ray": "2.54.0",
    "sglang": "0.5.9",
    "torch": "2.9.1+cu129",
    "transformers": "4.57.1",
    "vllm": "0.11.2",
    "watchfiles": "1.2.0",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    print(f"python={platform.python_version()}")
    mismatches = []
    for package, expected in EXPECTED.items():
        actual = importlib.metadata.version(package)
        print(f"{package}={actual}")
        if actual != expected:
            mismatches.append(f"{package}: expected {expected}, got {actual}")

    for module in ("PIL", "qwen_vl_utils", "ray", "sglang", "transformers",
                   "vllm"):
        importlib.import_module(module)
    from vllm import LLM, SamplingParams
    if not LLM or not SamplingParams:
        mismatches.append("vLLM public API import failed")

    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.backends.cudnn.version={torch.backends.cudnn.version()}")
    print(f"torch.cuda.nccl.version={torch.cuda.nccl.version()}")
    print(f"user={getpass.getuser()}")
    print(f"cuda.available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda.device={torch.cuda.get_device_name(0)}")
        print(f"cuda.capability={torch.cuda.get_device_capability(0)}")
        torch.ones(1, device="cuda").mul_(2)
    elif args.require_gpu:
        mismatches.append("CUDA GPU is required but unavailable")

    if mismatches:
        raise SystemExit("\n".join(mismatches))


if __name__ == "__main__":
    main()
