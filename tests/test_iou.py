"""
Tests for 3D Oriented Bounding Box module.

Validates IoU computation, transformations and edge cases.
"""

import math
import numpy as np
import pytest

from cvat_tool.iou import OBB3D, VolumeIntersectionCalculator, _GeometryConstants


# Test constants
UNIT_CUBE_SCALE = (2.0, 2.0, 2.0)
RECT_BOX_SCALE = (2.0, 3.0, 4.0)
ORIGIN = (0.0, 0.0, 0.0)
TOLERANCE = 1e-6


# ============================================================================
# Helper Functions
# ============================================================================

def _rt_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Builds a 4x4 rigid transformation matrix."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _rotz(theta: float) -> np.ndarray:
    """Rotation matrix around Z axis."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rotx(theta: float) -> np.ndarray:
    """Rotation matrix around X axis."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _roty(theta: float) -> np.ndarray:
    """Rotation matrix around Y axis."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rot3d(rx, ry, rz):
    """Combined rotation around all three axes (ZYX order)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler('zyx', [rz, ry, rx]).as_matrix()


# ============================================================================
# Tests for OBB3D class
# ============================================================================

class TestOBB3DBasics:
    """Test suite for basic OBB3D functionality."""

    @pytest.mark.parametrize(
        "scale",
        [UNIT_CUBE_SCALE, RECT_BOX_SCALE],
        ids=["unit_cube", "rectangular"],
    )
    def test_scaled_axis_aligned_vertices(self, scale):
        """Validates generation of axis-aligned box vertices."""
        scale_arr = np.array(scale)
        box = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), scale_arr)
        
        # Check number of points
        assert box.keypoints.shape == (_GeometryConstants.KEYPOINT_COUNT, 3)
        
        # Verify center is at origin
        center = box.keypoints[0, :]
        np.testing.assert_allclose(center, [0, 0, 0], atol=TOLERANCE)

    def test_from_pose_and_size_euler_angles(self):
        """Validates box creation from Euler angles."""
        euler = np.array([0.1, 0.2, 0.3])
        box = OBB3D.from_pose_and_size(euler, np.zeros(3), np.ones(3))
        
        assert box.keypoints.shape == (_GeometryConstants.KEYPOINT_COUNT, 3)

    def test_from_pose_and_size_rotation_matrix(self):
        """Validates box creation from rotation matrix."""
        rotation = _rotz(math.pi / 4)
        box = OBB3D.from_pose_and_size(rotation, np.zeros(3), np.ones(3))
        
        assert box.keypoints.shape == (_GeometryConstants.KEYPOINT_COUNT, 3)

    def test_from_pose_and_size_rejects_invalid_rotation(self):
        """Verifies that incorrect rotation format raises error."""
        invalid_rotation = np.array([1, 2])  # Wrong size
        
        with pytest.raises(ValueError, match="Invalid rotation format"):
            OBB3D.from_pose_and_size(invalid_rotation, np.zeros(3), np.ones(3))

    def test_volume_unit_cube(self):
        """Validates volume computation for unit cube."""
        box = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), np.ones(3))
        
        np.testing.assert_allclose(box.volume, 1.0, rtol=TOLERANCE)

    def test_volume_rectangular(self):
        """Validates volume computation for rectangular box."""
        scale = np.array([2.0, 3.0, 4.0])
        box = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), scale)
        
        expected = 2.0 * 3.0 * 4.0
        np.testing.assert_allclose(box.volume, expected, rtol=TOLERANCE)

    def test_rotation_property(self):
        """Validates extraction of rotation matrix."""
        rotation = _rotz(math.pi / 6)
        box = OBB3D.from_pose_and_size(rotation, np.zeros(3), np.ones(3))
        
        np.testing.assert_allclose(box.rotation_matrix, rotation, atol=TOLERANCE)

    def test_position_property(self):
        """Validates extraction of position vector."""
        position = np.array([1.0, 2.0, 3.0])
        box = OBB3D.from_pose_and_size(np.eye(3), position, np.ones(3))
        
        np.testing.assert_allclose(box.position, position, atol=TOLERANCE)

    def test_dimensions_property(self):
        """Validates extraction of dimensions."""
        dimensions = np.array([1.5, 2.5, 3.5])
        box = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), dimensions)
        
        np.testing.assert_allclose(box.dimensions, dimensions, rtol=TOLERANCE)

    def test_transform(self):
        """Validates transformation application to box."""
        box = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), np.ones(3))
        
        transform = _rt_transform(_rotz(math.pi / 4), np.array([1, 2, 3]))
        transformed = box.transform(transform)
        
        assert transformed.keypoints.shape == (_GeometryConstants.KEYPOINT_COUNT, 3)
        np.testing.assert_allclose(
            transformed.position, [1, 2, 3], atol=TOLERANCE
        )

    def test_contains_point_inside(self):
        """Validates point inside box detection."""
        box = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), np.ones(3))
        
        assert box.contains_point(np.array([0.0, 0.0, 0.0]))  # Center
        assert box.contains_point(np.array([0.4, 0.4, 0.4]))  # Inside

    def test_contains_point_outside(self):
        """Validates point outside box detection."""
        box = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), np.ones(3))
        
        assert not box.contains_point(np.array([1.0, 0.0, 0.0]))  # Outside boundary
        assert not box.contains_point(np.array([10, 10, 10]))  # Far away

    def test_sample_random_point_generates_points_inside(self):
        """Validates that sample generates points inside box."""
        box = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), np.ones(3))
        
        for _ in range(10):
            point = box.sample_random_point()
            assert box.contains_point(point)


# ============================================================================
# Tests for VolumeIntersectionCalculator
# ============================================================================

class TestVolumeIntersectionCalculator:
    """Test suite for intersection metrics."""

    def test_identical_boxes_iou_is_one(self):
        """Validates that IoU of identical boxes equals 1."""
        box1 = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), np.ones(3))
        box2 = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), np.ones(3))
        
        calculator = VolumeIntersectionCalculator(box1, box2)
        iou_value = calculator.calculate_iou()
        
        np.testing.assert_allclose(iou_value, 1.0, rtol=TOLERANCE)

    def test_non_overlapping_boxes_iou_is_zero(self):
        """Validates that IoU of non-overlapping boxes equals 0."""
        box1 = OBB3D.from_pose_and_size(
            np.eye(3), np.array([0, 0, 0]), np.ones(3)
        )
        box2 = OBB3D.from_pose_and_size(
            np.eye(3), np.array([10, 10, 10]), np.ones(3)
        )
        
        calculator = VolumeIntersectionCalculator(box1, box2)
        iou_value = calculator.calculate_iou()
        
        np.testing.assert_allclose(iou_value, 0.0, atol=TOLERANCE)

    def test_partial_overlap_iou(self):
        """Validates IoU for partially overlapping boxes."""
        box1 = OBB3D.from_pose_and_size(
            np.eye(3), np.array([0, 0, 0]), np.ones(3)
        )
        box2 = OBB3D.from_pose_and_size(
            np.eye(3), np.array([0.5, 0, 0]), np.ones(3)
        )
        
        calculator = VolumeIntersectionCalculator(box1, box2)
        iou_value = calculator.calculate_iou()
        
        # Intersection must be > 0 and < 1
        assert 0.0 < iou_value < 1.0

    def test_iou_symmetry(self):
        """Validates that IoU(A,B) = IoU(B,A)."""
        box1 = OBB3D.from_pose_and_size(
            np.eye(3), np.array([0, 0, 0]), np.ones(3)
        )
        box2 = OBB3D.from_pose_and_size(
            _rotz(math.pi / 4), np.array([0.3, 0.3, 0]), np.ones(3)
        )
        
        calc1 = VolumeIntersectionCalculator(box1, box2)
        calc2 = VolumeIntersectionCalculator(box2, box1)
        
        iou1 = calc1.calculate_iou()
        iou2 = calc2.calculate_iou()
        
        np.testing.assert_allclose(iou1, iou2, rtol=TOLERANCE)

    def test_iou_monte_carlo_approximation(self):
        """Validates that Monte Carlo approximation is close to exact IoU."""
        box1 = OBB3D.from_pose_and_size(
            np.eye(3), np.array([0, 0, 0]), np.ones(3)
        )
        box2 = OBB3D.from_pose_and_size(
            np.eye(3), np.array([0.5, 0, 0]), np.ones(3)
        )
        
        calculator = VolumeIntersectionCalculator(box1, box2)
        exact_iou = calculator.calculate_iou()
        sampled_iou = calculator.calculate_iou_monte_carlo(sample_count=50000)
        
        # Verify approximation is within 5%
        np.testing.assert_allclose(exact_iou, sampled_iou, rtol=0.05)


# ============================================================================
# Edge Cases and Special Scenarios
# ============================================================================

class TestEdgeCases:
    """Test suite for edge cases."""

    def test_zero_volume_box(self):
        """Validates box with zero volume (flat)."""
        scale = np.array([1.0, 1.0, 0.0])  # No depth
        box = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), scale)
        
        np.testing.assert_allclose(box.volume, 0.0, atol=TOLERANCE)

    def test_tiny_overlap_iou(self):
        """Validates IoU for very small intersection."""
        box1 = OBB3D.from_pose_and_size(
            np.eye(3), np.array([0, 0, 0]), np.ones(3)
        )
        box2 = OBB3D.from_pose_and_size(
            np.eye(3), np.array([0.99, 0, 0]), np.ones(3)
        )
        
        calculator = VolumeIntersectionCalculator(box1, box2)
        iou_value = calculator.calculate_iou()
        
        # Very small intersection but > 0
        assert 0.0 < iou_value < 0.1

    def test_rotated_boxes_iou(self):
        """Validates IoU for rotated boxes."""
        box1 = OBB3D.from_pose_and_size(
            np.eye(3), np.zeros(3), np.array([2.0, 2.0, 2.0])
        )
        box2 = OBB3D.from_pose_and_size(
            _rotz(math.pi / 4), np.zeros(3), np.array([2.0, 2.0, 2.0])
        )
        
        calculator = VolumeIntersectionCalculator(box1, box2)
        iou_value = calculator.calculate_iou()
        
        # Must have significant intersection
        assert 0.5 < iou_value <= 1.0

    def test_complex_realistic_scenario(self):
        """
        Realistic scenario test: two boxes with different sizes,
        both rotated in 3D, displaced in space.
        """
        # Box 1: rectangular, rotated 30°/20°/45° (X/Y/Z)
        rotation1 = _rot3d(np.deg2rad(30), np.deg2rad(20), np.deg2rad(45))
        position1 = np.array([1.0, 2.0, 1.0])
        scale1 = np.array([3.0, 2.0, 4.0])
        box1 = OBB3D.from_pose_and_size(rotation1, position1, scale1)
        
        # Box 2: cubic, rotated -15°/35°/-25° (X/Y/Z)
        rotation2 = _rot3d(np.deg2rad(-15), np.deg2rad(35), np.deg2rad(-25))
        position2 = np.array([2.0, 2.5, 0.5])
        scale2 = np.array([2.0, 2.0, 2.0])
        box2 = OBB3D.from_pose_and_size(rotation2, position2, scale2)
        
        # Verify box correctness
        np.testing.assert_allclose(box1.volume, 3.0 * 2.0 * 4.0, rtol=TOLERANCE)
        np.testing.assert_allclose(box2.volume, 2.0 * 2.0 * 2.0, rtol=TOLERANCE)
        
        # Compute IoU
        calculator = VolumeIntersectionCalculator(box1, box2)
        iou_value = calculator.calculate_iou()
        
        # Verify result validity
        assert 0.0 <= iou_value <= 1.0, f"IoU must be in [0,1], got: {iou_value}"
        
        # Verify intersection exists (boxes are close)
        assert iou_value > 0.0, "Must have intersection between boxes"
        
        # Verify IoU not too large (different sizes and positions)
        assert iou_value < 0.5, "IoU should not be too large for different boxes"
        
        # Verify symmetry
        calculator_reverse = VolumeIntersectionCalculator(box2, box1)
        iou_reverse = calculator_reverse.calculate_iou()
        np.testing.assert_allclose(iou_value, iou_reverse, rtol=TOLERANCE)


# ============================================================================
# Additional Custom Tests
# ============================================================================

class TestOBB3DExtra:
    def test_negative_dimensions(self):
        """Box with negative dimensions should behave as positive (absolute value)."""
        box = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), np.array([-2.0, -2.0, -2.0]))
        # Should be equivalent to positive size
        ref = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), np.array([2.0, 2.0, 2.0]))
        np.testing.assert_allclose(np.abs(box.dimensions), ref.dimensions, atol=1e-6)

    def test_large_box_contains_small_box(self):
        """Large box should contain the center of a small box inside it."""
        big = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), np.array([10, 10, 10]))
        small = OBB3D.from_pose_and_size(np.eye(3), np.array([1, 1, 1]), np.array([1, 1, 1]))
        assert big.contains_point(small.position)

class TestVolumeIntersectionCalculatorExtra:
    def test_touching_boxes_iou_zero(self):
        """Boxes that touch at a face but do not overlap should have IoU=0."""
        box1 = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), np.ones(3))
        box2 = OBB3D.from_pose_and_size(np.eye(3), np.array([1, 0, 0]), np.ones(3))
        calc = VolumeIntersectionCalculator(box1, box2)
        assert calc.calculate_iou() == 0.0

    def test_included_box_iou(self):
        """Box fully inside another: IoU = vol(small)/vol(big)."""
        big = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), np.array([4, 4, 4]))
        small = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), np.array([2, 2, 2]))
        calc = VolumeIntersectionCalculator(big, small)
        expected = small.volume / big.volume
        np.testing.assert_allclose(calc.calculate_iou(), expected, rtol=1e-6)

    def test_rotated_touching_boxes(self):
        """Rotated boxes that just touch at a corner should have IoU=0."""
        box1 = OBB3D.from_pose_and_size(np.eye(3), np.zeros(3), np.ones(3))
        box2 = OBB3D.from_pose_and_size(_rotz(np.pi/4), np.array([np.sqrt(2), 0, 0]), np.ones(3))
        calc = VolumeIntersectionCalculator(box1, box2)
        assert calc.calculate_iou() == 0.0

    def test_iou_with_random_boxes(self):
        """Random boxes: IoU always in [0,1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            pos1 = rng.uniform(-5, 5, 3)
            pos2 = rng.uniform(-5, 5, 3)
            size1 = rng.uniform(0.5, 3, 3)
            size2 = rng.uniform(0.5, 3, 3)
            rot1 = rng.uniform(-np.pi, np.pi, 3)
            rot2 = rng.uniform(-np.pi, np.pi, 3)
            box1 = OBB3D.from_pose_and_size(rot1, pos1, size1)
            box2 = OBB3D.from_pose_and_size(rot2, pos2, size2)
            calc = VolumeIntersectionCalculator(box1, box2)
            iou = calc.calculate_iou()
            assert 0.0 <= iou <= 1.0
