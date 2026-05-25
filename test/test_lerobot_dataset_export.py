import numpy as np

from multi_modal_data_collection.lerobot_dataset_export import LeRobotV21DatasetWriter


def test_compute_episode_stats_skips_string_scalars():
    writer = object.__new__(LeRobotV21DatasetWriter)

    rows = [
        {
            "timestamp": 0.0,
            "frame_index": 0,
            "franka_ee_pose_cmd_frame_id": "panda_link0",
            "state": [0.1, 0.2],
        },
        {
            "timestamp": 0.5,
            "frame_index": 1,
            "franka_ee_pose_cmd_frame_id": "panda_link0",
            "state": [0.3, 0.4],
        },
    ]

    stats = writer._compute_episode_stats(arrays={}, rows=rows)

    assert "franka_ee_pose_cmd_frame_id" not in stats
    assert stats["timestamp"]["min"] == [0.0]
    assert stats["timestamp"]["max"] == [0.5]
    assert stats["frame_index"]["min"] == [0]
    assert stats["frame_index"]["max"] == [1]
    assert np.allclose(stats["state"]["mean"], [0.2, 0.3])
