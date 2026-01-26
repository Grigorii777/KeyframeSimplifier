import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from iou import OBB3D, VolumeIntersectionCalculator

def _rot3d(rx, ry, rz):
    """Combined rotation around all three axes (ZYX order)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler('zyx', [rz, ry, rx]).as_matrix()


def plot_obb3d_list(boxes, ax=None, colors=None, alpha=0.3, edgecolor='k'):
    """
    Visualizes a list of OBB3D objects in matplotlib 3D Axes.
    :param boxes: list of OBB3D objects
    :param ax: matplotlib 3D Axes (if None, a new one is created)
    :param colors: list of colors for boxes (default: all blue)
    :param alpha: box transparency
    :param edgecolor: edge color
    :param iou_value: optional IoU value to display in the legend
    :return: matplotlib Figure and Axes
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    if colors is None:
        colors = ['C0'] * len(boxes)

    # Vertex indices for cube faces (OBB3D: 0 — center, 1–8 — corners)
    cube_faces = [
        [1, 2, 4, 3],  # bottom
        [5, 6, 8, 7],  # top
        [1, 2, 6, 5],  # front
        [3, 4, 8, 7],  # back
        [1, 3, 7, 5],  # left
        [2, 4, 8, 6],  # right
    ]
    for box, color in zip(boxes, colors):
        verts = box.keypoints  # (9, 3): 0 — center, 1–8 — corners
        faces = [[verts[i] for i in face] for face in cube_faces]
        pc = Poly3DCollection(faces, facecolor=color, alpha=alpha, edgecolor=edgecolor)
        ax.add_collection3d(pc)

    # Add legend with IoU if provided
    if len(boxes) == 2:
        calculator = VolumeIntersectionCalculator(boxes[0], boxes[1])
        iou_value = calculator.calculate_iou()
        if iou_value is not None:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors[0], edgecolor='k', label='Box 1'),
                Patch(facecolor=colors[1], edgecolor='k', label='Box 2'),
                Patch(facecolor='none', edgecolor='none', label=f'IoU = {iou_value:.4f}')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

    # Automatically set axis limits
    all_points = np.vstack([b.keypoints for b in boxes])
    for i in range(3):
        min_, max_ = all_points[:, i].min(), all_points[:, i].max()
        ax.set_xlim3d(min_, max_)
        ax.set_ylim3d(min_, max_)
        ax.set_zlim3d(min_, max_)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    return fig, ax

def ex_1():
    rotation1 = _rot3d(np.deg2rad(30), np.deg2rad(20), np.deg2rad(45))
    position1 = np.array([1.0, 2.0, 1.0])
    scale1 = np.array([3.0, 2.0, 4.0])
    box1 = OBB3D.from_pose_and_size(rotation1, position1, scale1)

    # Box 2: cubic, rotated -15°/35°/-25° (X/Y/Z)
    rotation2 = _rot3d(np.deg2rad(-15), np.deg2rad(35), np.deg2rad(-25))
    position2 = np.array([2.0, 2.5, 0.5])
    scale2 = np.array([2.0, 2.0, 2.0])
    box2 = OBB3D.from_pose_and_size(rotation2, position2, scale2)


    fig, ax = plot_obb3d_list([box1, box2], colors=['C0', 'C1'])
    # IoU legend is now handled inside plot_obb3d_list
    plt.show()

def ex_2():
    rotation1 = _rot3d(np.deg2rad(0), np.deg2rad(0), np.deg2rad(0))
    position1 = np.array([0.0, 0.0, 0.0])
    scale1 = np.array([1.0, 1.0, 1.0])
    box1 = OBB3D.from_pose_and_size(rotation1, position1, scale1)

    # Box 2: cubic, rotated -15°/35°/-25° (X/Y/Z)
    rotation2 = _rot3d(np.deg2rad(0), np.deg2rad(0), np.deg2rad(0))
    position2 = np.array([0.9, 0.9, 0.9])
    scale2 = np.array([1.0, 1.0, 1.0])
    box2 = OBB3D.from_pose_and_size(rotation2, position2, scale2)

    fig, ax = plot_obb3d_list([box1, box2], colors=['C0', 'C1'])
    # IoU legend is now handled inside plot_obb3d_list
    plt.show()

if __name__ == "__main__":
    ex_2()