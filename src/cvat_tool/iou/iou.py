"""3D OBB IoU calculation."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull


class _GeometryConstants:
    KEYPOINT_COUNT = 9
    FACE_INDICES = np.array([
        [5, 6, 8, 7], [1, 3, 4, 2], [3, 7, 8, 4],
        [1, 2, 6, 5], [2, 4, 8, 6], [1, 5, 7, 3]
    ])


# Plane classification
_EPS, _POS, _ZERO, _NEG = 0.000001, 1, 0, -1


class OBB3D:
    """3D oriented bounding box."""

    def __init__(self, kp=None):
        self._kp = kp if kp is not None else self._cube(np.ones(3))
        self._r = self._p = self._d = self._tf = self._vol = None

    def contains_point(self, pt):
        inv_tf = np.linalg.inv(self.transform_matrix)
        loc = inv_tf[:3, :3] @ pt + inv_tf[:3, 3]
        hd = self.dimensions / 2
        return all(abs(loc[i]) <= hd[i] for i in range(3))

    def __len__(self):
        return _GeometryConstants.KEYPOINT_COUNT

    def transform(self, T):
        if T.shape != (4, 4):
            raise ValueError('4x4 transformation matrix required')
        return OBB3D.from_pose_and_size(T[:3, :3] @ self.rotation_matrix,
                                        T[:3, 3] + T[:3, :3] @ self.position,
                                        self.dimensions)

    def sample_random_point(self):
        loc = np.random.uniform(-0.5, 0.5, 3) * self.dimensions
        return self.rotation_matrix @ loc + self.position
    
    @property
    def transform_matrix(self):
        if self._tf is None:
            self._tf = np.eye(4)
            self._tf[:3, :3], self._tf[:3, 3] = self.rotation_matrix, self.position
        return self._tf

    @property
    def keypoints(self):
        return self._kp
    
    @property
    def volume(self):
        if self._vol is None:
            self._vol = np.prod(self.dimensions)
        return self._vol
    
    @property
    def rotation_matrix(self):
        if self._r is None:
            v = self._kp[1:9] - self._kp[0]
            ax_x = (v[4] + v[5] + v[6] + v[7] - v[0] - v[1] - v[2] - v[3]) / 4
            ax_y = (v[2] + v[3] + v[6] + v[7] - v[0] - v[1] - v[4] - v[5]) / 4
            ax_z = (v[1] + v[3] + v[5] + v[7] - v[0] - v[2] - v[4] - v[6]) / 4
            self._r = np.column_stack([ax_x / np.linalg.norm(ax_x),
                                       ax_y / np.linalg.norm(ax_y),
                                       ax_z / np.linalg.norm(ax_z)])
        return self._r
    
    @classmethod
    def from_pose_and_size(cls, rot, pos, dim):
        if rot.size == 3:
            rot = R.from_rotvec(rot.flatten()).as_matrix()
        elif rot.size != 9:
            raise ValueError('Invalid rotation format')
        c = cls._cube(dim)
        k = (rot @ c.T).T + pos
        return cls(kp=k)
    
    @property
    def position(self):
        if self._p is None:
            self._p = self._kp[0].copy()
        return self._p
    
    def __repr__(self):
        return 'OBB3D:' + ''.join(f' [p{i}: {p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]' 
                                  for i, p in enumerate(self._kp))
    
    @property
    def dimensions(self):
        if self._d is None:
            v = self._kp[1:9] - self._kp[0]
            self._d = np.array([
                np.linalg.norm((v[4] + v[5] + v[6] + v[7] - v[0] - v[1] - v[2] - v[3]) / 4),
                np.linalg.norm((v[2] + v[3] + v[6] + v[7] - v[0] - v[1] - v[4] - v[5]) / 4),
                np.linalg.norm((v[1] + v[3] + v[5] + v[7] - v[0] - v[2] - v[4] - v[6]) / 4)
            ])
        return self._d

    @staticmethod
    def _cube(s):
        h = s / 2
        return np.array([[0, 0, 0], [-h[0], -h[1], -h[2]], [-h[0], -h[1], h[2]],
                        [-h[0], h[1], -h[2]], [-h[0], h[1], h[2]], [h[0], -h[1], -h[2]],
                        [h[0], -h[1], h[2]], [h[0], h[1], -h[2]], [h[0], h[1], h[2]]])


class VolumeIntersectionCalculator:
    """Calculate IoU for 3D OBBs."""

    def __init__(self, box_a, box_b):
        self._a, self._b, self._pts = box_a, box_b, []

    def _pt_vs_plane(self, pt, plane_pt, direction, axis):
        dist = direction * (pt[axis] - plane_pt[axis])
        return _POS if dist > _EPS else (_NEG if dist < -_EPS else _ZERO)

    def _edge_plane_isect(self, plane_pt, v1, v2, axis):
        t = (v2[axis] - plane_pt[axis]) / (v2[axis] - v1[axis])
        return t * v1 + (1 - t) * v2

    def _edge_plane_isect(self, plane_coord, v1, v2, axis):
        t = (v2[axis] - plane_coord) / (v2[axis] - v1[axis])
        return t * v1 + (1 - t) * v2
    
    def calculate_iou(self):
        self._pts = []
        self._extract(self._a, self._b)
        self._extract(self._b, self._a)
        
        if not self._pts:
            return 0.0
        
        try:
            hull = ConvexHull(np.array(self._pts))
            i_vol = hull.volume
        except:
            return 0.0
        
        u_vol = self._a.volume + self._b.volume - i_vol
        return i_vol / u_vol if u_vol > 1e-10 else 0.0
    
    def _extract(self, src, tpl):
        inv = np.linalg.inv(src.transform_matrix)
        src_loc = src.transform(inv)
        tpl_loc = tpl.transform(inv)
        
        # Process each face
        for f_id in range(len(_GeometryConstants.FACE_INDICES)):
            f_verts = _GeometryConstants.FACE_INDICES[f_id]
            poly = [tpl_loc.keypoints[f_verts[i]] for i in range(4)]
            clipped = self._clip_box(src_loc, poly)
            for loc_pt in clipped:
                world_pt = src.rotation_matrix @ loc_pt + src.position
                self._pts.append(world_pt)
        
        # Add contained vertices
        for i in range(9):
            if src_loc.contains_point(tpl_loc.keypoints[i]):
                world_pt = src.rotation_matrix @ tpl_loc.keypoints[i] + src.position
                self._pts.append(world_pt)

    def _clip_ax(self, poly, plane_pt, direction, axis):
        if len(poly) <= 1:
            return []
        
        out, all_on = [], True
        n = len(poly)
        
        for i in range(n):
            prev, curr = poly[(i - 1) % n], poly[i]
            prev_cls = self._pt_vs_plane(prev, plane_pt, direction, axis)
            curr_cls = self._pt_vs_plane(curr, plane_pt, direction, axis)
            
            if curr_cls == _NEG:
                all_on = False
                if prev_cls == _POS:
                    out.append(self._edge_plane_isect(plane_pt, prev, curr, axis))
                elif prev_cls == _ZERO and (not out or not np.array_equal(out[-1], prev)):
                    out.append(prev)
            elif curr_cls == _POS:
                all_on = False
                if prev_cls == _NEG:
                    out.append(self._edge_plane_isect(plane_pt, prev, curr, axis))
                elif prev_cls == _ZERO and (not out or not np.array_equal(out[-1], prev)):
                    out.append(prev)
                out.append(curr)
            else:  # ON
                if prev_cls != _ZERO:
                    out.append(curr)
        
        return poly if all_on else out
  
    def _clip_box(self, box, poly):
        w = poly
        for ax in range(3):
            w = self._clip_ax(w, box.keypoints[1], 1, ax)
            if not w:
                return []
            w = self._clip_ax(w, box.keypoints[8], -1, ax)
            if not w:
                return []
        return w

    def calculate_iou_monte_carlo(self, sample_count=10000):
        pts_a = [self._a.sample_random_point() for _ in range(sample_count)]
        in_b = sum(1 for p in pts_a if self._b.contains_point(p))
        est_a = self._a.volume * in_b
        
        pts_b = [self._b.sample_random_point() for _ in range(sample_count)]
        in_a = sum(1 for p in pts_b if self._a.contains_point(p))
        est_b = self._b.volume * in_a
        
        i_est = (est_a + est_b) / 2
        u_est = (self._a.volume + self._b.volume) * sample_count - i_est
        return i_est / u_est

