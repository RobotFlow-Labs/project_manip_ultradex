from anima_manip_ultradex.data.replay_buffer import (
    ReplayRecord,
    ReplayShardManifest,
    read_manifest,
    write_manifest,
)


def test_replay_manifest_roundtrip(tmp_path) -> None:
    records = [
        ReplayRecord(
            scene_points_path="shard/points_0001.npy",
            action=[0.0] * 36,
            strategy="pinch",
            object_id="bowl",
            stage="grasp",
            success=True,
        ),
        ReplayRecord(
            scene_points_path="shard/points_0002.npy",
            action=[0.1] * 36,
            strategy="bimanual",
            object_id="bowl",
            stage="lift",
            success=True,
        ),
    ]
    manifest = ReplayShardManifest.from_records("shard-0001", records)
    path = write_manifest(manifest, tmp_path / "manifest.json")
    loaded = read_manifest(path)

    assert loaded.shard_id == "shard-0001"
    assert loaded.num_frames == 2
    assert loaded.supported_strategies() == {"pinch", "bimanual"}
