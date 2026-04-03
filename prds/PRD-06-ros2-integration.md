# PRD-06: ROS2 Integration

> Module: MANIP-ULTRADEX | Priority: P1  
> Depends on: PRD-03, PRD-05  
> Status: ⬜ Not started

## Objective

Integrate MANIP-ULTRADEX into the ANIMA ROS2 stack so the policy can subscribe to scene point clouds and publish joint-space control outputs for both UR5e arms and XHands.

## Context (from paper)

The paper runs on a dual-arm hardware stack with RGB-D sensing. ANIMA requires a ROS2 bridge that expresses those signals as reusable topics and launch assets rather than direct Python calls.

## Acceptance Criteria

- [ ] A ROS2 node subscribes to point-cloud inputs and publishes dual-arm plus dual-hand action messages.
- [ ] Launch files bind camera, robot, and checkpoint parameters cleanly.
- [ ] A fake-node integration test confirms end-to-end message flow.
- [ ] Test: `uv run pytest tests/test_ros2_bridge.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_manip_ultradex/ros2/messages.py` | internal ROS message helpers | — | ~60 |
| `src/anima_manip_ultradex/ros2/policy_node.py` | ROS2 node | §VI.B embodiment | ~140 |
| `launch/manip_ultradex.launch.py` | launch composition | — | ~80 |
| `tests/test_ros2_bridge.py` | node-level tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs

- `/manip_ultradex/scene_pointcloud`
- `/manip_ultradex/robot_state`

### Outputs

- `/manip_ultradex/arm_action`
- `/manip_ultradex/hand_action`

### Algorithm

```python
class UltraDexPolicyNode(Node):
    def on_pointcloud(self, msg):
        action = self.runner.predict(msg.points)
        self.publish_actions(action)
```

## Dependencies

```toml
rclpy = "*"
sensor_msgs = "*"
std_msgs = "*"
```

## Data Requirements

| Asset | Size | Path | Download |
|---|---|---|---|
| ROS2 checkpoint | TBD | `/mnt/forge-data/models/manip-ultradex/policy/latest.ckpt` | produced locally |

## Test Plan

```bash
uv run pytest tests/test_ros2_bridge.py -v
```

## References

- Paper: §VI.B
- Depends on: PRD-03, PRD-05
- Feeds into: PRD-07
