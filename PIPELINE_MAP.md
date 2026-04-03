# MANIP-ULTRADEX — Pipeline Map

## Paper Pipeline → PRD Mapping

| Paper Component | Paper Section | Our File | PRD |
|---|---|---|---|
| Bimanual grasp pose definition and grasp wrench formulation | §III | `src/anima_manip_ultradex/grasp/types.py` | PRD-01 |
| Optimization-based grasp synthesis objective | §IV.A | `src/anima_manip_ultradex/grasp/bodex_adapter.py` | PRD-02 |
| Preferred-grasp ranking by SE(3) distance | §IV.A | `src/anima_manip_ultradex/grasp/selection.py` | PRD-02 |
| Four-stage demonstration generation | §IV.B | `src/anima_manip_ultradex/data/demo_generator.py` | PRD-02 |
| SAPIEN scene initialization and observation rendering | Fig. 2, §IV | `src/anima_manip_ultradex/sim/scene_env.py` | PRD-01 |
| Synthetic robot point-cloud augmentation | §IV.B and public env implementation | `src/anima_manip_ultradex/data/pointcloud_merge.py` | PRD-03 |
| PointNet++ point encoder | §V.B | `src/anima_manip_ultradex/policy/point_encoder.py` | PRD-02 |
| Decoder-only transformer backbone | §V.A–§V.C | `src/anima_manip_ultradex/policy/transformer.py` | PRD-02 |
| Unidirectional attention from action queries to scene tokens | §V.C, Fig. 4 | `src/anima_manip_ultradex/policy/action_queries.py` | PRD-02 |
| Bounded Gaussian action head | §V.C | `src/anima_manip_ultradex/policy/action_head.py` | PRD-02 |
| Checkpointed inference runner | §V and deployment adaptation | `src/anima_manip_ultradex/inference/runner.py` | PRD-03 |
| Simulation benchmark over 600 objects | §VI.A, Table I | `scripts/evaluate_sim.py` | PRD-04 |
| Real-world benchmark over 25 objects | §VI.B, Table III | `scripts/evaluate_real.py` | PRD-04 |
| FastAPI serving layer | ANIMA serving requirement | `src/anima_manip_ultradex/api/app.py` | PRD-05 |
| ROS2 bridge for dual-arm policy outputs | ANIMA integration requirement | `src/anima_manip_ultradex/ros2/policy_node.py` | PRD-06 |
| Export, validation, and release packaging | ANIMA production requirement | `scripts/export_policy.py` | PRD-07 |

## Data Flow

1. Object assets and robot URDFs are loaded into SAPIEN and cuRobo-aware world models.
2. BODex-based optimization synthesizes 500 candidate grasps for the chosen grasp strategy.
3. IK reachability, collision checks, and SE(3) ranking select the preferred grasp.
4. Motion planning generates the four-stage demonstration trajectory: pregrasp, grasp, squeeze, lift.
5. Scene point clouds are rendered, merged with imaged robot point clouds, cropped, and FPS-downsampled to 2,048 points.
6. The point encoder maps the 2,048-point scene into 256 scene tokens.
7. Four action-query tokens attend to scene tokens through unidirectional attention.
8. Arm and hand action heads emit bounded stochastic control commands for both manipulators.
9. Evaluation harnesses compute simulation and real-world success rates against the paper benchmarks.

## Public Repo Coverage vs ANIMA Work

| Area | Public Repo Coverage | ANIMA Work Required |
|---|---|---|
| Grasp synthesis | Present | Wrap and harden |
| Motion planning rollout | Present | Wrap and test |
| Policy training code | Absent | Rebuild from paper |
| Dataset serialization | Absent | Design and implement |
| Inference/export | Absent | Design and implement |
| API serving | Absent | Build |
| ROS2 integration | Absent | Build |
| Production validation | Absent | Build |

## Key Interfaces

| Interface | Shape / Contract | Producer | Consumer |
|---|---|---|---|
| `ScenePointCloud` | `Tensor[B, 2048, 3]` | `data/pointcloud_merge.py` | `policy/point_encoder.py` |
| `SceneTokens` | `Tensor[B, 256, D_scene]` | `policy/point_encoder.py` | `policy/transformer.py` |
| `ActionQueries` | `Tensor[B, 4, D_model]` | `policy/action_queries.py` | `policy/transformer.py` |
| `DualArmHandAction` | `Tensor[B, 36]` split into `2x6 + 2x12` | `policy/action_head.py` | `inference/runner.py` |
| `GraspCandidateSet` | `Tensor[G, H, 3, 19]` where `H in {1,2}` | `grasp/bodex_adapter.py` | `data/demo_generator.py` |

## Build Sequence

1. PRD-01 establishes package naming, config, types, and reference wrappers.
2. PRD-02 reconstructs the paper method and data-generation pipeline.
3. PRD-03 makes the model callable for offline inference and export.
4. PRD-04 validates paper-level reproduction targets.
5. PRD-05 through PRD-07 make the module deployable inside the ANIMA stack.
