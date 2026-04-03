from anima_manip_ultradex.config import load_module_config


def test_default_config_values() -> None:
    cfg = load_module_config()
    assert cfg.project.codename == "MANIP-ULTRADEX"
    assert cfg.project.paper_arxiv == "2603.05312"
    assert cfg.project.python_version == "3.11"
    assert cfg.paper.policy_input_points == 2048
    assert cfg.paper.action_query_tokens == 4


def test_path_registry_points_to_local_assets() -> None:
    cfg = load_module_config()
    assert cfg.paper_pdf.exists()
    assert cfg.reference_repo_root.exists()
    assert cfg.bowl_mesh.exists()
