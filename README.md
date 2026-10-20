# DRL deadline-aware scheduling of inference tasks on Kubernetes (MATD3)

Adapted from the HS-DRL `K8sEnv` / `PodDeploymentManager`.

## Layout
- `config.py`            nodes, task types/deadlines, reward, MATD3 hyper-parameters (edit first)
- `timing.py`            online pWCET estimator (empirical quantile or EVT/GPD)
- `environment/rt_env.py` gymnasium env (N agents, masked bids, delayed team reward)
- `environment/backends.py` SimBackend (training) and K8sBackend (real pods)
- `environment/pod_manager.py` creates finite task pods, measures real latency from container timestamps
- `agents/matd3.py`      MATD3 (per-node actors, centralised twin critic); `agents/baselines.py`
- `train.py`, `evaluate.py`, `profile_cluster.py`, `setup_cluster.sh`, `task_image/`

## Workflow
1. `pip install -r requirements.txt` ; `./setup_cluster.sh` (k3d, 1 server + 3 agents)
2. `python profile_cluster.py`  -> writes `calibration.json` (replaces the placeholder simulator parameters)
3. `python train.py --episodes 300`            (simulator, minutes on CPU)
4. `python evaluate.py --backend sim`           then `python evaluate.py --backend k8s --episodes 5`
5. Optional fine-tune on the cluster: `python train.py --backend k8s --episodes 20 --load ckpt/matd3_best.pt`
