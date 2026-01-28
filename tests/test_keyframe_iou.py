
import numpy as np
from cvat_tool.keyframe_handler import Keyframe
from cvat_tool.keyframe_iou import choose_keyframes_iou


def test_two_frames_only():
    """Test with only two frames"""
    keyframes = [
        Keyframe(0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        Keyframe(1, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
    ]
    result = choose_keyframes_iou(keyframes, iou_threshold=0.8)
    assert result == [0, 1]


def test_three_frames_straight_line():
    """Test with three frames in straight line - middle frame should be skipped with high threshold"""
    keyframes = [
        Keyframe(0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        Keyframe(1, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        Keyframe(2, np.array([2.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
    ]
    # With high threshold, middle frame should match interpolation well
    result = choose_keyframes_iou(keyframes, iou_threshold=0.9)
    assert 0 in result
    assert 2 in result
    # Middle frame should be skipped due to high IoU with interpolation
    assert 1 not in result


def test_moving_and_scaling_box():
    """Test with box that moves and changes size"""
    keyframes = [
        Keyframe(0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        Keyframe(1, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([1.5, 1.0, 1.0])),
        Keyframe(2, np.array([2.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])),
        Keyframe(3, np.array([3.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([2.5, 1.0, 1.0])),
        Keyframe(4, np.array([4.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([3.0, 1.0, 1.0])),
    ]
    result = choose_keyframes_iou(keyframes, iou_threshold=0.8)
    # Should include first and last
    assert 0 in result
    assert 4 in result
    # Should have some intermediate keyframes due to scaling
    assert len(result) >= 2


def test_rotating_box():
    """Test with rotating box"""
    keyframes = [
        Keyframe(0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])),
        Keyframe(1, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.5]), np.array([2.0, 1.0, 1.0])),
        Keyframe(2, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([2.0, 1.0, 1.0])),
        Keyframe(3, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.5]), np.array([2.0, 1.0, 1.0])),
    ]
    # Rotation causes IoU to drop, so more keyframes should be kept
    result_strict = choose_keyframes_iou(keyframes, iou_threshold=0.95)
    result_relaxed = choose_keyframes_iou(keyframes, iou_threshold=0.5)
    
    assert 0 in result_strict
    assert 3 in result_strict
    assert 0 in result_relaxed
    assert 3 in result_relaxed
    
    # Stricter threshold should keep more keyframes
    assert len(result_strict) >= len(result_relaxed)


def test_sudden_position_change():
    """Test with sudden position jump"""
    keyframes = [
        Keyframe(0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        Keyframe(1, np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        Keyframe(2, np.array([5.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),  # Sudden jump
        Keyframe(3, np.array([5.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        Keyframe(4, np.array([6.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
    ]
    result = choose_keyframes_iou(keyframes, iou_threshold=0.8)
    # Should detect the sudden jump and keep frame 2
    assert 0 in result
    assert 4 in result
    assert 2 in result
    # Frames 1 and 3 should likely be skipped (smooth motion before and after jump)
    # Note: This depends on IoU behavior, but at least one should be skipped
    assert len(result) < 5, "Should simplify and skip some intermediate frames"


def test_threshold_comparison():
    """Test that lower threshold results in fewer keyframes"""
    keyframes = [
        Keyframe(i, np.array([float(i), 0.0, 0.0]), np.array([0.0, 0.0, float(i) * 0.1]), 
                np.array([1.0 + float(i) * 0.1, 1.0, 1.0]))
        for i in range(10)
    ]
    
    result_low = choose_keyframes_iou(keyframes, iou_threshold=0.3)
    result_high = choose_keyframes_iou(keyframes, iou_threshold=0.9)
    
    # Lower threshold = more simplification = fewer keyframes
    assert len(result_low) <= len(result_high)
    # Both should always include first and last
    assert 0 in result_low
    assert 9 in result_low
    assert 0 in result_high
    assert 9 in result_high
    # With low threshold, should skip most intermediate frames
    assert len(result_low) < 10, "Low threshold should simplify the sequence"


    