import numpy as np
import pytest
from cvat_tool.keyframe_handler import KeyframeHandler
from cvat_tool.dto import Keyframe
from unittest.mock import Mock


def create_mock_shape_with_attributes(frame_id, attributes=None, outside=False):
    """Create a mock shape object with attributes for testing."""
    shape = Mock()
    shape.frame = frame_id
    shape.outside = outside
    shape.points = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # position + rotation + scale
    
    # Create mock attributes
    shape.attributes = []
    if attributes:
        for spec_id, value in attributes.items():
            attr = Mock()
            attr.spec_id = spec_id
            attr.value = value
            shape.attributes.append(attr)
    
    return shape


def make_keyframes(frames_attrs):
    """
    Utility to create a list of Keyframe objects for tests.
    frames_attrs: list of tuples (frame_id, pos, rot, scale, attributes)
    pos, rot, scale can be None (defaults will be used)
    attributes: dict or None
    """
    def arr(val, default):
        return np.array(val) if val is not None else np.array(default)
    kfs = []
    for fa in frames_attrs:
        frame_id = fa[0]
        pos = arr(fa[1], [0.0, 0.0, 0.0])
        rot = arr(fa[2], [0.0, 0.0, 0.0])
        scale = arr(fa[3], [1.0, 1.0, 1.0])
        attributes = fa[4] if len(fa) > 4 and fa[4] is not None else {}
        kfs.append(Keyframe(frame_id, pos, rot, scale, attributes=attributes))
    return kfs


def test_find_attribute_change_no_attributes():
    """Test with keyframes that have no attributes."""
    handler = KeyframeHandler()
    keyframes = make_keyframes([
        (0, None, None, None, {}),
        (1, [1.0, 0.0, 0.0], None, None, {}),
        (2, [2.0, 0.0, 0.0], None, None, {}),
    ])
    result = handler.find_attribute_change_keyframes(keyframes)
    assert len(result) == 0, "Should find no changes when there are no attributes"


def test_find_attribute_change_constant_attributes():
    """Test with keyframes that have constant attributes."""
    handler = KeyframeHandler()
    keyframes = make_keyframes([
        (0, None, None, None, {1: "true"}),
        (1, [1.0, 0.0, 0.0], None, None, {1: "true"}),
        (2, [2.0, 0.0, 0.0], None, None, {1: "true"}),
    ])
    result = handler.find_attribute_change_keyframes(keyframes)
    assert len(result) == 0, "Should find no changes when attributes are constant"


def test_find_attribute_change_single_change():
    """Test with a single attribute change."""
    handler = KeyframeHandler()
    keyframes = make_keyframes([
        (0, None, None, None, {1: "false"}),
        (1, [1.0, 0.0, 0.0], None, None, {1: "false"}),
        (2, [2.0, 0.0, 0.0], None, None, {1: "true"}),  # Change here
        (3, [3.0, 0.0, 0.0], None, None, {1: "true"}),
    ])
    result = handler.find_attribute_change_keyframes(keyframes)
    assert result == {2}, f"Expected {{2}}, got {result}"


def test_find_attribute_change_multiple_changes():
    """Test with multiple attribute changes."""
    handler = KeyframeHandler()
    keyframes = make_keyframes([
        (0, None, None, None, {1: "false"}),
        (1, [1.0, 0.0, 0.0], None, None, {1: "true"}),   # Change 1
        (2, [2.0, 0.0, 0.0], None, None, {1: "true"}),
        (3, [3.0, 0.0, 0.0], None, None, {1: "false"}),  # Change 2
        (4, [4.0, 0.0, 0.0], None, None, {1: "true"}),   # Change 3
    ])
    result = handler.find_attribute_change_keyframes(keyframes)
    assert result == {1, 3, 4}, f"Expected {{1, 3, 4}}, got {result}"


def test_find_attribute_change_multiple_attributes():
    """Test with multiple attributes per keyframe."""
    handler = KeyframeHandler()
    keyframes = make_keyframes([
        (0, None, None, None, {1: "false", 2: "open"}),
        (1, [1.0, 0.0, 0.0], None, None, {1: "false", 2: "closed"}),  # attribute 2 changed
        (2, [2.0, 0.0, 0.0], None, None, {1: "true", 2: "closed"}),   # attribute 1 changed
        (3, [3.0, 0.0, 0.0], None, None, {1: "true", 2: "closed"}),
    ])
    result = handler.find_attribute_change_keyframes(keyframes)
    assert result == {1, 2}, f"Expected {{1, 2}}, got {result}"


def test_find_attribute_change_attribute_added():
    """Test when a new attribute is added."""
    handler = KeyframeHandler()
    keyframes = make_keyframes([
        (0, None, None, None, {1: "false"}),
        (1, [1.0, 0.0, 0.0], None, None, {1: "false", 2: "open"}),  # New attribute added
        (2, [2.0, 0.0, 0.0], None, None, {1: "false", 2: "open"}),
    ])
    result = handler.find_attribute_change_keyframes(keyframes)
    assert result == {1}, f"Expected {{1}}, got {result}"


def test_find_attribute_change_attribute_removed():
    """Test when an attribute is removed."""
    handler = KeyframeHandler()
    keyframes = make_keyframes([
        (0, None, None, None, {1: "false", 2: "open"}),
        (1, [1.0, 0.0, 0.0], None, None, {1: "false"}),  # Attribute 2 removed
        (2, [2.0, 0.0, 0.0], None, None, {1: "false"}),
    ])
    result = handler.find_attribute_change_keyframes(keyframes)
    assert result == {1}, f"Expected {{1}}, got {result}"


def test_simplifying_with_attribute_changes():
    """Test that simplifying preserves keyframes with attribute changes."""
    handler = KeyframeHandler()
    # Create keyframes where positions form a straight line (can be simplified)
    # But attribute changes at frame 2
    keyframes = make_keyframes([
        (0, None, None, None, {1: "false"}),
        (1, [1.0, 0.0, 0.0], None, None, {1: "false"}),
        (2, [2.0, 0.0, 0.0], None, None, {1: "true"}),  # Attribute changes here
        (3, [3.0, 0.0, 0.0], None, None, {1: "true"}),
        (4, [4.0, 0.0, 0.0], None, None, {1: "true"}),
    ])
    # Simplify with high IoU threshold (would normally skip middle frames)
    result = handler.simplifying(keyframes, iou_threshold=0.95, fields=None, auto_percent=0)
    # Frame 2 must be kept because of attribute change
    result_frame_ids = [kf.frame_id for kf in result]
    assert 2 in result_frame_ids, f"Frame 2 with attribute change must be preserved. Got frames: {result_frame_ids}"
    # First and last frames should always be kept
    assert 0 in result_frame_ids
    assert 4 in result_frame_ids


def test_simplifying_multiple_attribute_changes():
    """Test simplifying with multiple attribute changes."""
    handler = KeyframeHandler()
    keyframes = make_keyframes([
        (0, None, None, None, {1: "false"}),
        (1, [1.0, 0.0, 0.0], None, None, {1: "true"}),   # Change
        (2, [2.0, 0.0, 0.0], None, None, {1: "true"}),
        (3, [3.0, 0.0, 0.0], None, None, {1: "false"}),  # Change
        (4, [4.0, 0.0, 0.0], None, None, {1: "false"}),
        (5, [5.0, 0.0, 0.0], None, None, {1: "true"}),   # Change
    ])
    result = handler.simplifying(keyframes, iou_threshold=0.95, fields=None, auto_percent=0)
    result_frame_ids = [kf.frame_id for kf in result]
    # All frames with attribute changes must be kept
    assert 1 in result_frame_ids, "Frame 1 with attribute change must be preserved"
    assert 3 in result_frame_ids, "Frame 3 with attribute change must be preserved"
    assert 5 in result_frame_ids, "Frame 5 with attribute change must be preserved"


def test_prepare_keyframes_from_shapes_with_attributes():
    """Test that prepare_keyframes_from_shapes correctly extracts attributes."""
    handler = KeyframeHandler()
    
    shapes = [
        create_mock_shape_with_attributes(0, attributes={1: "false", 2: "open"}),
        create_mock_shape_with_attributes(1, attributes={1: "true", 2: "closed"}),
        create_mock_shape_with_attributes(2, attributes={}),
    ]
    
    keyframes = handler.prepare_keyframes_from_shapes(shapes)
    
    assert len(keyframes) == 3
    assert keyframes[0].attributes == {1: "false", 2: "open"}
    assert keyframes[1].attributes == {1: "true", 2: "closed"}
    assert keyframes[2].attributes == {}


def test_get_simplified_frame_ids_with_attribute_changes():
    """Test get_simplified_frame_ids preserves frames with attribute changes."""
    handler = KeyframeHandler()
    
    shapes = [
        create_mock_shape_with_attributes(0, attributes={1: "false"}),
        create_mock_shape_with_attributes(1, attributes={1: "false"}),
        create_mock_shape_with_attributes(2, attributes={1: "true"}),  # Change
        create_mock_shape_with_attributes(3, attributes={1: "true"}),
        create_mock_shape_with_attributes(4, attributes={1: "false"}), # Change
        create_mock_shape_with_attributes(5, attributes={1: "false"}),
    ]
    
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.95, auto_percent=0)
    
    # Frames with attribute changes must be included
    assert 2 in result, "Frame 2 with attribute change must be preserved"
    assert 4 in result, "Frame 4 with attribute change must be preserved"


def test_attribute_changes_with_outside_frames():
    """Test that attribute changes work correctly with outside frames."""
    handler = KeyframeHandler()
    
    shapes = [
        create_mock_shape_with_attributes(0, attributes={1: "false"}, outside=False),
        create_mock_shape_with_attributes(1, attributes={1: "true"}, outside=False),  # Change
        create_mock_shape_with_attributes(2, attributes={1: "true"}, outside=True),   # Outside
        create_mock_shape_with_attributes(3, attributes={1: "false"}, outside=False), # Change
        create_mock_shape_with_attributes(4, attributes={1: "false"}, outside=False),
    ]
    
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.95, auto_percent=0)
    
    # Outside frame must be included
    assert 2 in result, "Outside frame must be preserved"
    # Frames with attribute changes must be included
    assert 1 in result, "Frame 1 with attribute change must be preserved"
    assert 3 in result, "Frame 3 with attribute change must be preserved"


def test_edge_case_single_keyframe():
    """Test with a single keyframe."""
    handler = KeyframeHandler()
    keyframes = make_keyframes([
        (0, None, None, None, {1: "false"}),
    ])
    result = handler.find_attribute_change_keyframes(keyframes)
    assert len(result) == 0, "Single keyframe should have no attribute changes"


def test_edge_case_empty_keyframes():
    """Test with empty keyframes list."""
    handler = KeyframeHandler()
    keyframes = []
    result = handler.find_attribute_change_keyframes(keyframes)
    assert len(result) == 0, "Empty list should have no attribute changes"
