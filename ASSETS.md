# MANIP-ULTRADEX — Asset Manifest

## Paper
- Title: UltraDexGrasp: Learning Universal Dexterous Grasping for Bimanual Robots with Synthetic Data
- ArXiv: 2603.05312
- Authors: Sizhe Yang, Yiman Xie, Zhixuan Liang, Yang Tian, Jia Zeng, Dahua Lin, Jiangmiao Pang
- Project page: https://yangsizhe.github.io/ultradexgrasp/

## Status: ALMOST

The repo originally pointed at `arXiv:2503.13300`, but `papers/2503.13300_UltraDexGrasp.pdf` is an unrelated motion-generation paper.
The correct UltraDexGrasp paper is stored locally at `papers/2603.05312_UltraDexGrasp.pdf` and is the source of truth for this module.

## Reference Implementations

| Component | Source | Local Path | Status |
|---|---|---|---|
| Public rollout and synthetic data pipeline | https://github.com/InternRobotics/UltraDexGrasp | `repositories/UltraDexGrasp/` | DONE |
| Grasp synthesis backend | https://github.com/yangsizhe/BODex_api | `third_party/BODex_api` in upstream install flow | MISSING |
| Motion planning backend | https://github.com/NVlabs/curobo | `third_party/curobo` in upstream install flow | MISSING |
| Point-cloud geometry ops | https://github.com/facebookresearch/pytorch3d | `third_party/pytorch3d` in upstream install flow | MISSING |

## Pretrained Weights

| Model | Size | Source | Path on Server | Status |
|---|---|---|---|---|
| UltraDexGrasp policy checkpoint | Not disclosed | Paper and public repo do not publish weights | `/mnt/forge-data/models/manip-ultradex/policy/latest.ckpt` | MISSING |
| ONNX export | Not disclosed | To be produced from local training run | `/mnt/forge-data/models/manip-ultradex/policy/latest.onnx` | MISSING |
| MLX inference export | Not disclosed | To be produced from local conversion pass | `/mnt/forge-data/models/manip-ultradex/policy/latest_mlx/` | MISSING |

## Datasets

| Dataset | Size | Split | Source | Path | Status |
|---|---|---|---|---|---|
| UltraDexGrasp-20M trajectories | 20M frames across 1,000 objects | train/val/test split must be recreated locally | Generated with the paper pipeline from DexGraspNet-derived assets | `/mnt/forge-data/datasets/manip-ultradex/ultradexgrasp_20m/` | MISSING |
| DexGraspNet object assets | 1,000 selected objects for this paper | source asset pool | https://arxiv.org/abs/2210.02697 and DexGraspNet release assets | `/mnt/forge-data/datasets/dexgraspnet/selected_1000/` | MISSING |
| Simulation benchmark object list | 600 objects | seen and unseen test groups | Defined in paper Section VI.A | `/mnt/forge-data/datasets/manip-ultradex/benchmarks/sim_600/` | MISSING |
| Real-world benchmark object list | 25 objects | eval only | Defined in paper Section VI.B | `/mnt/forge-data/datasets/manip-ultradex/benchmarks/real_25/` | MISSING |
| Bowl demo asset from public repo | 1 object | smoke test only | Vendored with public repo | `repositories/UltraDexGrasp/asset/object_mesh/bowl/` | DONE |

## Hardware and Embodiment

| Asset | Spec | Path / Notes | Status |
|---|---|---|---|
| Simulation robot | 2x UR5e + 2x 12-DoF XHand | public repo `env/base_env.py` and paper Section VI.A | DONE |
| Real robot | 2x UR5e + 2x 12-DoF XHand | paper Section VI.B | AVAILABLE |
| RGB-D sensors | 2x Azure Kinect DK, eye-on-base | paper Section VI.B | AVAILABLE |
| Synthetic robot point clouds | 6 camera viewpoints via SAPIEN renderer | `repositories/UltraDexGrasp/env/util/synthetic_pc_util.py` | DONE |
| CUDA workstation | NVIDIA GPU required for BODex and cuRobo | build target | REQUIRED |
| MLX workstation | Apple Silicon optional for ANIMA inference export | downstream deployment target | OPTIONAL |

## Hyperparameters and Pipeline Constants (from paper)

| Param | Value | Paper Section |
|---|---|---|
| candidate_grasps_per_object | 500 | §IV.A |
| pregrasp_offset_m | 0.1 | §IV.B |
| lift_target_m | 0.2 | §IV.B |
| lift_success_height_m | 0.17 | §IV.B |
| lift_hold_time_s | 1.0 | §IV.B |
| policy_input_points | 2048 | §V.B |
| point_group_knn | 32 | §V.B |
| abstraction_output_points | 256 | §V.B |
| action_distribution | bounded Gaussian via truncated normal | §V.C |
| real_control_frequency_hz | 10 | §VI.B |
| optimizer | Not specified in paper | — |
| batch_size | Not specified in paper | — |
| learning_rate | Not specified in paper | — |
| epochs | Not specified in paper | — |

## Expected Metrics (from paper)

| Benchmark | Metric | Paper Value | Our Target |
|---|---|---|---|
| Simulation benchmark, overall | success rate | 84.0% | >= 79.0% on reproduced benchmark before real deployment |
| Simulation benchmark, unseen objects | success rate | 83.4% | >= 78.0% |
| Simulation benchmark, seen small | success rate | 78.8% | >= 73.0% |
| Simulation benchmark, seen medium | success rate | 84.3% | >= 79.0% |
| Simulation benchmark, seen large | success rate | 90.4% | >= 85.0% |
| Simulation benchmark, unseen small | success rate | 76.9% | >= 71.0% |
| Simulation benchmark, unseen medium | success rate | 85.8% | >= 80.0% |
| Simulation benchmark, unseen large | success rate | 87.5% | >= 82.0% |
| Real-world benchmark, overall | success rate | 81.2% | >= 75.0% |
| Real-world benchmark, small | success rate | 72.0% | >= 65.0% |
| Real-world benchmark, medium | success rate | 82.2% | >= 76.0% |
| Real-world benchmark, large | success rate | 89.3% | >= 84.0% |

## Key Tensor and Schema Targets

| Item | Shape | Source |
|---|---|---|
| policy input point cloud | `Tensor[B, 2048, 3]` | §V.B and public env FPS path |
| point encoder stage-1 groups | `Tensor[B, 2048, 32, 3+C]` | §V.B |
| point encoder stage-2 tokens | `Tensor[B, 256, D_scene]` | §V.B |
| action query tokens | `Tensor[B, 4, D_model]` | inferred from Fig. 4 dual-arm plus dual-hand heads |
| arm actions | `Tensor[B, 2, 6]` | inferred from 2x UR5e in §VI |
| hand actions | `Tensor[B, 2, 12]` | inferred from 2x XHand in §VI |
| synthesized grasp candidates, unimanual | `Tensor[G, 1, 3, 19]` | public `util/bodex_util.py` |
| synthesized grasp candidates, bimanual | `Tensor[G, 2, 3, 19]` | public `util/bodex_util.py` |

## Immediate Asset Actions

1. Remove or quarantine the incorrect `papers/2503.13300_UltraDexGrasp.pdf` reference from project docs.
2. Materialize a local manifest for the 1,000 DexGraspNet-derived objects used to build UltraDexGrasp-20M.
3. Decide whether ANIMA reproduction will vendor BODex/cuRobo directly or wrap the upstream repos as optional extras.
4. Recreate the missing training split metadata, because the public repo only ships rollout/data-generation code.
