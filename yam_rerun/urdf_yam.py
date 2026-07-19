"""Dual-arm YAM URDF loading + forward kinematics for Rerun 0.34.1.

Loads ``YAM/i2rt/i2rt/robot_models/yam/yam.urdf`` twice (entity/frame prefixes
``left_arm`` and ``right_arm``), anchors the two root frames apart in space, and
maps the MolmoAct2 14-D joint state vector onto both trees per timestep.

The URDF has exactly six revolute joints per arm (``joint1``..``joint6``,
base to wrist) and **no gripper joint** — the gripper is a separate linear
actuator (i2rt ``GripperType.LINEAR_4310``, normalized by i2rt's
``GripperAdapter``), so state indices 6 and 13 are carried in the vector but
ignored by FK here.

Follows the pattern of the so100-hackathon reference (``urdf_arm.py``) and the
``rerun-urdf`` skill in ``.agents/skills/``: ``log_urdf_to_recording`` for the
static model, a static ``Transform3D`` with ``child_frame`` to anchor each
URDF's frame island into the entity tree, and per-frame
``UrdfJoint.compute_transform`` for FK.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import rerun as rr

REPO_ROOT = Path(__file__).resolve().parents[1]
YAM_URDF_PATH = REPO_ROOT / "YAM" / "i2rt" / "i2rt" / "robot_models" / "yam" / "yam.urdf"

# ---------------------------------------------------------------------------
# 14-D state-vector contract.
#
# Ground truth: YAM/molmoact_to_lerobot_v30.py :: STATE_DIM_NAMES — the
# converter that produced the released allenai/MolmoAct2-BimanualYAM datasets
# (norm_tag "yam_dual_molmoact2"). Ordering is
#   [left_joint1..left_joint6, left_gripper,
#    right_joint1..right_joint6, right_gripper]
# i.e. state = concat(left 7-vector, right 7-vector), each 7-vector being the
# i2rt joint state [q1..q6 (rad), gripper (normalized linear actuator; no
# joint in yam.urdf)]. URDF joint names are exactly "joint1".."joint6".
# ---------------------------------------------------------------------------
STATE_DIM_NAMES: tuple[str, ...] = (
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_gripper",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_gripper",
)
STATE_DIM = len(STATE_DIM_NAMES)  # 14

LEFT_ARM = "left_arm"
RIGHT_ARM = "right_arm"

#: Per-arm URDF joint name -> index into the 14-D state vector.
JOINT_STATE_INDEX: dict[str, dict[str, int]] = {
    LEFT_ARM: {f"joint{i}": i - 1 for i in range(1, 7)},
    RIGHT_ARM: {f"joint{i}": 7 + i - 1 for i in range(1, 7)},
}
#: Gripper index into the 14-D state vector, per arm (present in the state,
#: absent from yam.urdf — FK skips it).
GRIPPER_STATE_INDEX: dict[str, int] = {LEFT_ARM: 6, RIGHT_ARM: 13}


def _ensure_meshes_resolvable(urdf_path: Path = YAM_URDF_PATH) -> None:
    """Make ``package://assets/*.stl`` mesh URIs in yam.urdf resolvable.

    Rerun's URDF mesh resolver handles ``package://<pkg>/<file>`` by searching
    ``ROS_PACKAGE_PATH`` (then ``AMENT_PREFIX_PATH``) for a directory named
    ``<pkg>``. Pointing ``ROS_PACKAGE_PATH`` at the URDF's own directory makes
    the sibling ``assets/`` dir resolvable — no patched copy of the URDF
    needed. Resolution happens lazily at ``log_urdf_to_recording`` time, so
    setting the env var in-process before logging is sufficient (verified on
    rerun-sdk 0.34.1). Appends rather than overwrites in case a real ROS
    environment is active (e.g. the robot workstation).
    """
    pkg_dir = str(urdf_path.parent)
    existing = os.environ.get("ROS_PACKAGE_PATH", "")
    if pkg_dir not in existing.split(os.pathsep):
        os.environ["ROS_PACKAGE_PATH"] = (
            f"{existing}{os.pathsep}{pkg_dir}" if existing else pkg_dir
        )


@dataclass
class YamArm:
    """One YAM arm: a parsed URDF tree plus its slot in the 14-D state vector."""

    name: str
    """State-vector slot: "left_arm" or "right_arm" (keys JOINT_STATE_INDEX)."""
    prefix: str
    """Entity/frame prefix; defaults to ``name``. Ghost instances use e.g.
    "ghost/left_arm" so a second robot renders in the same 3D view."""
    label: str
    """Text of the floating name tag; empty string suppresses the tag."""
    tree: rr.urdf.UrdfTree
    joints: list[rr.urdf.UrdfJoint]
    """Revolute joints ordered joint1..joint6 (base to wrist)."""
    state_indices: list[int]
    """Index into the 14-D state vector, aligned with ``joints``."""
    gripper_state_index: int
    translation: tuple[float, float, float]
    """World position of this arm's base (root frames anchor here)."""
    visual_geometries_path: str
    """Entity path of the visual meshes (what a 3D blueprint view needs)."""
    collision_geometries_path: str
    """Entity path of the collision meshes (for hiding in blueprints)."""

    @classmethod
    def create(
        cls,
        name: str,
        *,
        translation: tuple[float, float, float],
        urdf_path: Path = YAM_URDF_PATH,
        prefix: str | None = None,
        label: str | None = None,
    ) -> "YamArm":
        if name not in JOINT_STATE_INDEX:
            raise ValueError(f"name must be one of {sorted(JOINT_STATE_INDEX)}, got {name!r}")
        prefix = prefix if prefix is not None else name
        _ensure_meshes_resolvable(urdf_path)
        tree = rr.urdf.UrdfTree.from_file_path(
            urdf_path, entity_path_prefix=prefix, frame_prefix=f"{prefix}/"
        )
        # yam.urdf declares joints wrist-first (joint6..joint1); sort by name
        # ("joint1".."joint6" sorts correctly lexically) into base-to-wrist order.
        joints = sorted(
            (j for j in tree.joints() if j.joint_type == "revolute"),
            key=lambda j: j.name,
        )
        expected = sorted(JOINT_STATE_INDEX[name])
        if [j.name for j in joints] != expected:
            raise RuntimeError(
                f"{urdf_path} revolute joints {[j.name for j in joints]} != expected {expected}"
            )
        return cls(
            name=name,
            prefix=prefix,
            label=label if label is not None else name,
            tree=tree,
            joints=joints,
            state_indices=[JOINT_STATE_INDEX[name][j.name] for j in joints],
            gripper_state_index=GRIPPER_STATE_INDEX[name],
            translation=translation,
            visual_geometries_path=f"{prefix}/{tree.name}/visual_geometries",
            collision_geometries_path=f"{prefix}/{tree.name}/collision_geometries",
        )

    def visual_geometry_paths(self) -> list[str]:
        """Entity path of every visual mesh (for tinting/blueprint content filters)."""
        links = [self.tree.root_link()] + [self.tree.get_joint_child(j) for j in self.tree.joints()]
        return [path for link in links for path in self.tree.get_visual_geometry_paths(link)]

    def tint(self, rec: rr.RecordingStream, rgba: tuple[int, int, int, int]) -> None:
        """Override every visual mesh's albedo (static). Used to render ghost
        (predicted-trajectory) instances translucent/recolored."""
        for path in self.visual_geometry_paths():
            rec.log(path, rr.Asset3D.from_fields(albedo_factor=rgba), static=True)

    def log_static(self, rec: rr.RecordingStream) -> None:
        """(Re-)log the URDF meshes + fixed transforms and anchor the root frame.

        Static data is keyed by recording id, so call once per recording/take.
        """
        self.tree.log_urdf_to_recording(rec)
        # The URDF's named frames form an island: anchor its root frame
        # ("<prefix>/base_link") into the entity tree, or nothing renders (both
        # arms would otherwise collapse onto the world origin).
        root_frame = f"{self.prefix}/{self.tree.root_link().name}"
        rec.log(
            self.prefix,
            rr.Transform3D(translation=self.translation, child_frame=root_frame),
            static=True,
        )
        # Floating name tag above the base: makes left/right unambiguous from any
        # orbit angle (a mirrored viewpoint otherwise reads as swapped arms).
        if self.label:
            x, y, z = self.translation
            rec.log(
                f"labels/{self.prefix}",
                rr.Points3D([(x, y, z + 0.55)], labels=[self.label], radii=[0.003]),
                static=True,
            )

    def joint_angle(self, joint_index: int, value: float) -> float:
        """Clamp a state-vector angle (rad) to the URDF joint's limits.

        Clamping here (rather than compute_transform(clamp=True)) avoids a
        warning per out-of-limit sample flooding stdout.
        """
        joint = self.joints[joint_index]
        return min(max(value, joint.limit_lower), joint.limit_upper)

    def log_joints(self, rec: rr.RecordingStream, state: "Sequence[float]") -> None:
        """Apply this arm's slice of a 14-D state vector as joint transforms.

        Uses the current time set on ``rec`` (call ``rec.set_time`` first).
        """
        for joint_index, (joint, state_index) in enumerate(
            zip(self.joints, self.state_indices)
        ):
            angle = self.joint_angle(joint_index, float(state[state_index]))
            rec.log(f"{self.prefix}/joint_transforms", joint.compute_transform(angle))


@dataclass
class DualYam:
    """Both YAM arms, anchored apart along world y (left arm at +y, z-up)."""

    left: YamArm
    right: YamArm

    @classmethod
    def create(
        cls,
        *,
        spacing: float = 0.6,
        urdf_path: Path = YAM_URDF_PATH,
        prefix: str = "",
        label: str | None = None,
    ) -> "DualYam":
        """``prefix``/``label`` build a second instance (e.g. ``prefix="ghost/"``,
        ``label=""``) that renders alongside the default one in the same 3D view."""
        half = spacing / 2.0
        return cls(
            left=YamArm.create(
                LEFT_ARM, translation=(0.0, half, 0.0), urdf_path=urdf_path,
                prefix=f"{prefix}{LEFT_ARM}" if prefix else None, label=label,
            ),
            right=YamArm.create(
                RIGHT_ARM, translation=(0.0, -half, 0.0), urdf_path=urdf_path,
                prefix=f"{prefix}{RIGHT_ARM}" if prefix else None, label=label,
            ),
        )

    def tint(self, rec: rr.RecordingStream, rgba: tuple[int, int, int, int]) -> None:
        for arm in self.arms:
            arm.tint(rec, rgba)

    @property
    def arms(self) -> tuple[YamArm, YamArm]:
        return (self.left, self.right)

    def log_static(self, rec: rr.RecordingStream) -> None:
        for arm in self.arms:
            arm.log_static(rec)

    def log_state(self, rec: rr.RecordingStream, state: "Sequence[float]") -> None:
        """Apply one 14-D joint vector to both trees at the current timestep."""
        if len(state) != STATE_DIM:
            raise ValueError(f"expected a {STATE_DIM}-D state vector, got {len(state)}")
        for arm in self.arms:
            arm.log_joints(rec, state)
