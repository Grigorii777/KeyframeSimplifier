import numpy as np
from cvat_tool.keyframe_handler import KeyframeHandler
from cvat_tool.dto import Keyframe
from unittest.mock import Mock


def create_mock_shape(frame_id, outside=False):
    """Create a mock shape object for testing."""
    shape = Mock()
    shape.frame = frame_id
    shape.outside = outside
    shape.points = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # position + rotation + scale
    return shape


def test_no_outside_frames():
    """Test simplification with no outside frames."""
    handler = KeyframeHandler()
    shapes = [create_mock_shape(i, outside=False) for i in range(10)]
    
    # With high threshold, should simplify
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.9, auto_percent=0)
    
    # Should include first and last frames at minimum
    assert 0 in result
    assert 9 in result


def test_single_outside_in_middle():
    """Test with one outside frame in the middle."""
    handler = KeyframeHandler()
    shapes = [
        create_mock_shape(0, outside=False),
        create_mock_shape(1, outside=False),
        create_mock_shape(2, outside=False),
        create_mock_shape(3, outside=True),   # outside frame
        create_mock_shape(4, outside=False),
        create_mock_shape(5, outside=False),
    ]
    
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.9, auto_percent=0)
    
    # Outside frame must be included
    assert 3 in result
    # Should have segments [0,1,2] and [4,5]
    # First frame of each segment should be included
    assert 0 in result  # first frame of first segment
    assert 2 in result
    assert 4 in result  # first frame after outside
    assert 5 in result  # last frame of second segment


def test_multiple_outside_frames():
    """Test with multiple outside frames."""
    handler = KeyframeHandler()
    shapes = [
        create_mock_shape(0, outside=False),
        create_mock_shape(1, outside=False),
        create_mock_shape(2, outside=True),   # outside
        create_mock_shape(3, outside=False),
        create_mock_shape(4, outside=False),
        create_mock_shape(5, outside=True),   # outside
        create_mock_shape(6, outside=False),
        create_mock_shape(7, outside=False),
    ]
    
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.9, auto_percent=0)
    
    for i in range(8):
        assert i in result
    
def test_multiple_outside_frames2():
    """Test with multiple outside frames."""
    handler = KeyframeHandler()
    shapes = [
        create_mock_shape(0, outside=False),
        create_mock_shape(1, outside=False),
        create_mock_shape(2, outside=True),   # outside
        create_mock_shape(3, outside=False),
        create_mock_shape(4, outside=False),
        create_mock_shape(5, outside=True),   # outside
        create_mock_shape(6, outside=False),
    ]
    
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.9, auto_percent=0)
    
    for i in range(7):
        assert i in result

def test_multiple_outside_frames3():
    """Test with multiple outside frames."""
    handler = KeyframeHandler()
    shapes = [
        create_mock_shape(0, outside=False),
        create_mock_shape(1, outside=True), # outside
        create_mock_shape(2, outside=False),   
        create_mock_shape(3, outside=False),
        create_mock_shape(4, outside=True), # outside
        create_mock_shape(5, outside=False),   
    ]
    
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.9, auto_percent=0)
    
    for i in range(6):
        assert i in result


def test_outside_at_start():
    """Test with outside frame at the beginning."""
    handler = KeyframeHandler()
    shapes = [
        create_mock_shape(0, outside=True),   # outside at start
        create_mock_shape(1, outside=False),
        create_mock_shape(2, outside=False),
        create_mock_shape(3, outside=False),
    ]
    
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.9, auto_percent=0)
    
    # Outside frame must be included
    assert 0 in result
    # First frame after outside should be included
    assert 1 in result


def test_outside_at_end():
    """Test with outside frame at the end."""
    handler = KeyframeHandler()
    shapes = [
        create_mock_shape(0, outside=False),
        create_mock_shape(1, outside=False),
        create_mock_shape(2, outside=False),
        create_mock_shape(3, outside=True),   # outside at end
    ]
    
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.9, auto_percent=0)
    
    # Outside frame must be included
    assert 3 in result
    # First frame should be included
    assert 0 in result
    assert 2 in result


def test_consecutive_outside_frames():
    """Test with consecutive outside frames."""
    handler = KeyframeHandler()
    shapes = [
        create_mock_shape(0, outside=False),
        create_mock_shape(1, outside=False),
        create_mock_shape(2, outside=True),
        create_mock_shape(3, outside=True),
        create_mock_shape(4, outside=False),
        create_mock_shape(5, outside=False),
    ]
    
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.9, auto_percent=0)
    
    # Both outside frames must be included
    assert 2 in result
    assert 3 in result
    # First frame after last outside
    assert 4 in result

    assert 0 in result
    assert 1 in result
    assert 5 in result


def test_all_outside_frames():
    """Test when all frames are outside."""
    handler = KeyframeHandler()
    shapes = [
        create_mock_shape(0, outside=True),
        create_mock_shape(1, outside=True),
        create_mock_shape(2, outside=True),
    ]
    
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.9, auto_percent=0)
    
    # All outside frames must be included
    assert 0 in result
    assert 1 in result
    assert 2 in result


def test_segment_simplification():
    """Test that segments between outside frames are simplified independently."""
    handler = KeyframeHandler()
    
    # Create shapes with varying positions
    shapes = []
    for i in range(10):
        shape = Mock()
        shape.frame = i
        shape.outside = (i == 5)  # frame 5 is outside
        # Segment 1 (0-4): linear movement
        # Segment 2 (6-9): linear movement
        x_pos = float(i if i < 5 else i - 5)
        shape.points = [x_pos, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        shapes.append(shape)
    
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.8, auto_percent=0)
    
    # Outside frame must be included
    assert 5 in result
    # Should simplify both segments
    assert 0 in result  # first of segment 1
    assert 6 in result  # first after outside


def test_empty_segment_after_outside():
    """Test when there's an outside frame at the very end with no frames after it."""
    handler = KeyframeHandler()
    shapes = [
        create_mock_shape(0, outside=False),
        create_mock_shape(1, outside=False),
        create_mock_shape(2, outside=True),   # outside at end, no frames after
    ]
    
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.9, auto_percent=0)
    
    # Outside frame must be included
    assert 2 in result
    # First frame should be included
    assert 0 in result
    # Should not crash or have issues with empty segment after frame 2


def test_single_frame_segment():
    """Test with single frame segments between outside frames."""
    handler = KeyframeHandler()
    shapes = [
        create_mock_shape(0, outside=True),
        create_mock_shape(1, outside=False),  # single frame segment
        create_mock_shape(2, outside=True),
        create_mock_shape(3, outside=False),  # single frame segment
        create_mock_shape(4, outside=True),
    ]
    
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.9, auto_percent=0)
    
    # All outside frames
    assert 0 in result
    assert 2 in result
    assert 4 in result
    # Single frames should be included
    assert 1 in result
    assert 3 in result

def test_single_frame_segment2():
    """Test with single frame segments between outside frames."""
    handler = KeyframeHandler()
    shapes = [
        create_mock_shape(1, outside=False),  # single frame segment
        create_mock_shape(2, outside=True),
        create_mock_shape(3, outside=False),
        create_mock_shape(4, outside=False),
        create_mock_shape(5, outside=False),
    ]
    
    result = handler.get_simplified_frame_ids(shapes, fields=None, iou_threshold=0.9, auto_percent=0)
    

    assert 1 in result
    assert 2 in result
    assert 3 in result
    assert 4 not in result
    assert 5 in result
