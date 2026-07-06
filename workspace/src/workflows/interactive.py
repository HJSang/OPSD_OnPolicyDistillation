"""
Interactive development workflow for RLPilot.

This module provides workflows for interactive development environments,
including VSCode integration for debugging and development.
"""
import os
import subprocess
from time import sleep

import flytekit
from flytekit import Secret, task, workflow
from flytekitplugins.kfpytorch import PyTorch
from flytekitplugins.vscode import vscode

SECRET_GROUP = "codefetcher-secret"
SECRET_NAME = "github_ae_proxy_url"
CONTAINER_IMAGE = (
    "{{.image.mldev_verl_vllm_cu128_image}}"
    # "container-image-registry.corp.linkedin.com/temp/lifomo/lifomo-verl:202509040120"
)

def git_proxy_setup():
    """Set up git proxy configuration for GitHub access."""
    context = flytekit.current_context()
    git_proxy = context.secrets.get(SECRET_GROUP, SECRET_NAME)
    # Set up git proxy
    subprocess.run(["git", "config", "--global", "http.proxyAuthMethod", "basic"], check=True)
    subprocess.run(["git", "config", "--global", "http.proxy", git_proxy], check=True)
    os.environ["HTTP_PROXY"] = git_proxy
    os.environ["HTTPS_PROXY"] = git_proxy
    # For these domains, we can not use proxy
    os.environ["NO_PROXY"] = (
        "lerna.tools.corp.linkedin.com,artifactory.corp.linkedin.com,"
        "mlflow.grid1.ard.grid.linkedin.com"
    )


@task(
    enable_nfs=True,
    secret_requests=[Secret(group=SECRET_GROUP, key=SECRET_NAME)],
    task_config=PyTorch(num_workers=0),
    instance_type="h200_1",
    container_image=CONTAINER_IMAGE,
    proxy_as="coreaiopt",
    enable_identity_certs=True,
)
@vscode(
    enable=True,
    pre_execute=git_proxy_setup,
    max_idle_seconds=86400,
    port=8080,
)
def vscode_task():
    """VSCode task for interactive development."""
    pass


@workflow(namespace="training-coreai")
def vscode_workflow():
    """
    Workflow for launching an interactive VSCode development environment.
    
    This workflow creates a pod with VSCode server enabled, allowing developers
    to connect via the Kubernetes extension in VSCode for interactive debugging
    and development.
    """
    vscode_task()


@task(
    enable_nfs=True,
    secret_requests=[Secret(group=SECRET_GROUP, key=SECRET_NAME)],
    task_config=PyTorch(num_workers=0),
    instance_type="h100_2",
    container_image=CONTAINER_IMAGE,
    proxy_as="coreaiopt",
    enable_identity_certs=True,
)
@vscode(
    enable=False,
    pre_execute=git_proxy_setup,
    max_idle_seconds=86400,
    port=8080,
)
def mldev_h100_2_task():
    """H100_2 task for interactive development."""
    git_proxy_setup()
    import time
    time.sleep(10000000)


@workflow(namespace="training-coreai")
def mldev_h100_2_workflow():
    mldev_h100_2_task()


@task(
    enable_nfs=True,
    secret_requests=[Secret(group=SECRET_GROUP, key=SECRET_NAME)],
    task_config=PyTorch(num_workers=0),
    instance_type="h100_8",
    container_image=CONTAINER_IMAGE,
    proxy_as="coreaiopt",
    enable_identity_certs=True,
)
@vscode(
    enable=False,
    pre_execute=git_proxy_setup,
    max_idle_seconds=86400,
    port=8080,
)
def mldev_h100_8_task():
    """H100_8 task for interactive development."""
    git_proxy_setup()
    import time
    time.sleep(10000000)


@workflow(namespace="training-coreai")
def mldev_h100_8_workflow():
    mldev_h100_8_task()


@task(
    enable_nfs=True,
    secret_requests=[Secret(group=SECRET_GROUP, key=SECRET_NAME)],
    task_config=PyTorch(num_workers=0),
    instance_type="h200_2",
    container_image=CONTAINER_IMAGE,
    proxy_as="coreaiopt",
    enable_identity_certs=True,
)
@vscode(
    enable=False,
    pre_execute=git_proxy_setup,
    max_idle_seconds=86400,
    port=8080,
)
def mldev_h200_2_task():
    """H200_2 task for interactive development."""
    git_proxy_setup()
    import time
    time.sleep(10000000)


@workflow(namespace="training-coreai")
def mldev_h200_2_workflow():
    mldev_h200_2_task()


@task(
    enable_nfs=True,
    environment={
        "WEAVER_URLS": "weaver.prod-ltx1.atd.disco.linkedin.com:8998",
    },
    secret_requests=[Secret(group=SECRET_GROUP, key=SECRET_NAME)],
    task_config=PyTorch(num_workers=0),
    instance_type="h200_1",
    container_image=CONTAINER_IMAGE,
    proxy_as="coreaiopt",
    enable_identity_certs=True,
)
def dev_task():
    """Development task that keeps a pod running for SSH access."""
    while True:
        sleep(60)


@workflow(namespace="training-coreai")
def dev_workflow():
    """
    Workflow for launching a development pod with SSH access.
    
    This workflow creates a long-running pod that can be accessed via SSH
    for interactive development and debugging.
    """
    dev_task()
