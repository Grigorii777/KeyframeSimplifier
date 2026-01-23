from dataclasses import dataclass
import torch
from pytorch3d.ops import box3d_overlap

"""
!!!!! Doesn`t work good for MacOS and new python versions !!!!!!!!
"""

def _euler_zyx_to_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Angles in radians."""
    cr, sr = torch.cos(torch.tensor(roll)), torch.sin(torch.tensor(roll))
    cp, sp = torch.cos(torch.tensor(pitch)), torch.sin(torch.tensor(pitch))
    cy, sy = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))

    rx = torch.tensor([[1, 0, 0],
                       [0, cr, -sr],
                       [0, sr,  cr]], dtype=torch.float32)

    ry = torch.tensor([[ cp, 0, sp],
                       [  0, 1,  0],
                       [-sp, 0, cp]], dtype=torch.float32)

    rz = torch.tensor([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [ 0,   0, 1]], dtype=torch.float32)

    return rz @ ry @ rx


def _box9dof_to_corners(
    x: float, y: float, z: float,
    roll: float, pitch: float, yaw: float,
    sx: float, sy: float, sz: float,
) -> torch.Tensor:
    """
    Convert (center xyz, roll/pitch/yaw, full sizes sx/sy/sz) to 8 corners.
    Corner order MUST match PyTorch3D box3d_overlap docs.
    """
    # Full sizes -> half-extents
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

    # Local corners in PyTorch3D order:
    # [0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]
    corners_local = torch.tensor([
        [-hx, -hy, -hz],
        [ hx, -hy, -hz],
        [ hx,  hy, -hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [ hx,  hy,  hz],
        [-hx,  hy,  hz],
    ], dtype=torch.float32)

    R = _euler_zyx_to_matrix(roll, pitch, yaw)  # (3,3)
    center = torch.tensor([x, y, z], dtype=torch.float32)

    return corners_local @ R.T + center

@dataclass
class Box3D:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    sx: float
    sy: float
    sz: float


def iou_3d_from_9dof(a: Box3D, b: Box3D) -> float:
    """
    a/b keys: X,Y,Z, Roll,Pitch,Yaw, ScaleX,ScaleY,ScaleZ
    All angles are radians.
    """
    ca = _box9dof_to_corners(a.x, a.y, a.z,
                            a.roll, a.pitch, a.yaw,
                            a.sx, a.sy, a.sz).unsqueeze(0)
    cb = _box9dof_to_corners(b.x, b.y, b.z,
                            b.roll, b.pitch, b.yaw,
                            b.sx, b.sy, b.sz).unsqueeze(0)

    _, iou = box3d_overlap(ca, cb)  # expects (B, 8, 3) 
    return float(iou[0, 0].item())


if __name__ == "__main__":
    box1 = Box3D(x=0, y=0, z=0,
                 roll=0, pitch=0, yaw=0,
                 sx=2, sy=2, sz=2)
    box2 = Box3D(x=1, y=1, z=1,
                 roll=0, pitch=0, yaw=0,
                 sx=2, sy=2, sz=2)

    print("IOU:", iou_3d_from_9dof(box1, box2))
    