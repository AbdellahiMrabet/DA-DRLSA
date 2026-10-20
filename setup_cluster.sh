#!/usr/bin/env bash
# Lightweight 1-server + 3-agent k3d cluster sized for an 8 GB machine.
set -euo pipefail
k3d cluster create rt --servers 1 --agents 3 \
  --k3s-arg "--disable=traefik@server:0" \
  --k3s-arg "--disable=servicelb@server:0" \
  --k3s-arg "--disable=metrics-server@server:0"

# emulate heterogeneous nodes: must match NODE_PROFILES["cores"] in config.py
docker update --cpus 0.75 --memory 1g    --memory-swap 1g    k3d-rt-agent-0
docker update --cpus 1.25 --memory 1250m --memory-swap 1250m k3d-rt-agent-1
docker update --cpus 1.75 --memory 1250m --memory-swap 1250m k3d-rt-agent-2

docker build -t rt-task:latest task_image
k3d image import rt-task:latest -c rt
kubectl create namespace rt-bench --dry-run=client -o yaml | kubectl apply -f -
kubectl get nodes
