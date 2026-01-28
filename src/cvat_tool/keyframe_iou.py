from cvat_tool.keyframe_handler import Keyframe
from cvat_tool.iou.iou import OBB3D, VolumeIntersectionCalculator
import numpy as np
import pytest


def choose_keyframes_iou(keyframes: list[Keyframe], iou_threshold: float) -> list[int]:
    """Select minimal set of keyframes for track simplification using IoU metric.

    Uses an adaptive algorithm similar to Ramer-Douglas-Peucker to find
    the minimal set of keyframes needed to represent a track. A keyframe is kept
    if the IoU between the actual shape and interpolated shape is below the threshold
    (meaning they differ significantly).

    Args:
        keyframes: List of Keyframe objects with position, rotation, and scale
        iou_threshold: Minimum IoU threshold (0.0 to 1.0). Lower values mean more
                      simplification. If IoU < threshold, keyframe is kept.

    Returns:
        Sorted list of keyframe indices to keep

    Example:
        # Keep keyframes where IoU with interpolation is < 0.8
        indices = choose_keyframes_iou(keyframes, iou_threshold=0.8)
    """
    if len(keyframes) <= 2:
        return list(range(len(keyframes)))

    def interpolate_keyframe(kf1: Keyframe, kf2: Keyframe, alpha: float) -> Keyframe:
        """Linearly interpolate between two keyframes."""
        return Keyframe(
            frame_id=0,  # Not used for comparison
            position=kf1.position + alpha * (kf2.position - kf1.position),
            rotation=kf1.rotation + alpha * (kf2.rotation - kf1.rotation),
            scale=kf1.scale + alpha * (kf2.scale - kf1.scale),
        )

    def keyframe_to_obb(kf: Keyframe) -> OBB3D:
        """Convert Keyframe to OBB3D for IoU calculation."""
        return OBB3D.from_pose_and_size(
            rot=kf.rotation,
            pos=kf.position,
            dim=kf.scale
        )

    def calculate_iou(kf1: Keyframe, kf2: Keyframe) -> float:
        """Calculate IoU between two keyframes."""
        obb1 = keyframe_to_obb(kf1)
        obb2 = keyframe_to_obb(kf2)
        calculator = VolumeIntersectionCalculator(obb1, obb2)
        return calculator.calculate_iou()

    def process_range(start: int, stop: int) -> list[int]:
        """Recursively find keyframes in a subrange [start, stop]."""
        # Base case: adjacent frames, both are keyframes
        if stop - start <= 1:
            return [start, stop]

        start_kf = keyframes[start]
        stop_kf = keyframes[stop]

        # Find the frame with worst IoU (most different from interpolation)
        worst_idx = -1
        worst_iou = 1.0  # Best possible IoU

        for i in range(start + 1, stop):
            # Calculate interpolation parameter based on actual frame IDs
            actual_frame_id = keyframes[i].frame_id
            alpha = (actual_frame_id - start_kf.frame_id) / (stop_kf.frame_id - start_kf.frame_id)
            
            # Interpolate keyframe
            interpolated_kf = interpolate_keyframe(start_kf, stop_kf, alpha)
            
            # Calculate IoU between actual and interpolated
            actual_kf = keyframes[i]
            print(f"Frame {actual_kf.frame_id} (index={i}):")
            print(f"  Actual:       pos={actual_kf.position}, rot={actual_kf.rotation}, scale={actual_kf.scale}")
            print(f"  Interpolated: pos={interpolated_kf.position}, rot={interpolated_kf.rotation}, scale={interpolated_kf.scale}")
            iou_value = calculate_iou(actual_kf, interpolated_kf)
            print(f"  IoU = {iou_value}")
            
            # Track the worst IoU (lowest value = most different)
            if iou_value < worst_iou:
                worst_iou = iou_value
                worst_idx = i

        # If worst IoU is still above threshold, we can skip intermediate frames
        if worst_iou >= iou_threshold:
            return [start, stop]

        print("Keyframe: id =", worst_idx, "iou =", worst_iou)
        # Split at worst frame and recurse
        left_keyframes = process_range(start, worst_idx)
        right_keyframes = process_range(worst_idx, stop)

        # Merge results (worst_idx appears in both, so take left + right[1:])
        return left_keyframes + right_keyframes[1:]

    return process_range(0, len(keyframes) - 1)

