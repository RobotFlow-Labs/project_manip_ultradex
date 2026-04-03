from anima_manip_ultradex.config import load_module_config
from anima_manip_ultradex.grasp.bodex_adapter import BodexAdapter
from anima_manip_ultradex.planning.curobo_adapter import CuroboAdapter
from anima_manip_ultradex.sim.scene_env import SceneEnvAdapter


def test_reference_assets_exist() -> None:
    cfg = load_module_config()
    fixture_paths = SceneEnvAdapter(cfg).fixture_paths()
    assert fixture_paths["paper_pdf"].exists()
    assert fixture_paths["bowl_mesh"].exists()
    assert fixture_paths["reference_env"].exists()


def test_lazy_wrappers_report_missing_runtime_without_import_crash() -> None:
    cfg = load_module_config()
    bodex = BodexAdapter(cfg).availability()
    curobo = CuroboAdapter(cfg).availability()
    assert bodex["reference_file"] is True
    assert curobo["reference_file"] is True
