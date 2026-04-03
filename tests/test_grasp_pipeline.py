from anima_manip_ultradex.config import load_module_config
from anima_manip_ultradex.data.demo_generator import DemoGenerator
from anima_manip_ultradex.grasp.object_model import load_bowl_fixture
from anima_manip_ultradex.grasp.selection import rank_grasps_by_se3, select_preferred_grasp
from anima_manip_ultradex.grasp.types import GraspCandidate, GraspCandidateSpec, Pose7D


def _pose(x: float, y: float, z: float) -> Pose7D:
    return Pose7D(xyz=(x, y, z), wxyz=(1.0, 0.0, 0.0, 0.0))


def _candidate(x: float, strategy: str = "bimanual") -> GraspCandidate:
    return GraspCandidate(
        strategy=strategy,  # type: ignore[arg-type]
        object_id="bowl",
        num_hands=2,
        wrist_pose=_pose(x, 0.0, 0.0),
        hand_joints=[0.0] * 12,
        score=1.0 - x,
    )


def test_bowl_fixture_exists() -> None:
    cfg = load_module_config()
    bowl = load_bowl_fixture(cfg)
    assert bowl.exists()


def test_candidate_tensor_shape_contract() -> None:
    spec = GraspCandidateSpec(num_hands=2)
    assert spec.tensor_rank == 4
    assert spec.tensor_shape(500) == (500, 2, 3, 19)


def test_grasp_ranking_prefers_nearest_pose() -> None:
    reference_pose = _pose(0.0, 0.0, 0.0)
    far_candidate = _candidate(0.30)
    near_candidate = _candidate(0.02)

    ranked = rank_grasps_by_se3(reference_pose, [far_candidate, near_candidate])
    preferred = select_preferred_grasp(reference_pose, [far_candidate, near_candidate], "bimanual")

    assert ranked[0].candidate == near_candidate
    assert preferred == near_candidate


def test_demo_generator_emits_four_stage_rollout() -> None:
    cfg = load_module_config()
    demo = DemoGenerator(cfg).generate(_candidate(0.01))

    assert demo.stage_names() == ("pregrasp", "grasp", "squeeze", "lift")
    assert demo.stages[0].target_pose.xyz[2] > demo.stages[1].target_pose.xyz[2]
    assert demo.stages[-1].target_pose.xyz[2] > demo.stages[-2].target_pose.xyz[2]
