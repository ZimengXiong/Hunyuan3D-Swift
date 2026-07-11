"""Compact mesh renderer on the cr-faithful rasterizer (numpy), AA OFF.

Ports the reference MeshRender essentials needed by the paint pipeline:
mesh normalization, orthographic camera, and the normal / position control-map renders
(the geometry conditioning the multiview UNet consumes). No CUDA, no torch.
"""

from __future__ import annotations

import math
import os
import numpy as np

from .raster import cr_raster

# Rasterizer backend: 'auto' uses the Apple-GPU Metal kernel when available (bit-exact
# face-id vs the numpy reference, ~158x faster), else the numpy reference. Override via PAINT_RASTER.
_RASTER_BACKEND = os.environ.get("PAINT_RASTER", "auto")


def _rasterize(pos, idx, resolution):
    return cr_raster.rasterize_backend(pos, idx, resolution, backend=_RASTER_BACKEND)


def get_mv_matrix(elev, azim, camera_distance, center=None):
    elev = -elev
    azim += 90
    er, ar = math.radians(elev), math.radians(azim)
    cam = np.array([camera_distance * math.cos(er) * math.cos(ar),
                    camera_distance * math.cos(er) * math.sin(ar),
                    camera_distance * math.sin(er)])
    center = np.array([0, 0, 0]) if center is None else np.array(center)
    lookat = center - cam
    lookat /= np.linalg.norm(lookat)
    up = np.array([0, 0, 1.0])
    right = np.cross(lookat, up); right /= np.linalg.norm(right)
    up = np.cross(right, lookat); up /= np.linalg.norm(up)
    c2w = np.concatenate([np.stack([right, up, -lookat], axis=-1), cam[:, None]], axis=-1)
    w2c = np.zeros((4, 4), np.float32)
    w2c[:3, :3] = c2w[:3, :3].T
    w2c[:3, 3:] = -c2w[:3, :3].T @ c2w[:3, 3:]
    w2c[3, 3] = 1.0
    return w2c.astype(np.float32)


def get_ortho_proj(scale=1.2, near=0.0, far=2.0):
    l = -scale * 0.5; r = scale * 0.5; b = -scale * 0.5; t = scale * 0.5
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 2 / (r - l); m[1, 1] = 2 / (t - b); m[2, 2] = -2 / (far - near)
    m[0, 3] = -(r + l) / (r - l); m[1, 3] = -(t + b) / (t - b); m[2, 3] = -(far + near) / (far - near)
    return m


def mean_vertex_normals(n_verts, faces, face_normals):
    vn = np.zeros((n_verts, 3), np.float64)
    for k in range(3):
        np.add.at(vn, faces[:, k], face_normals)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    return (vn / np.clip(norm, 1e-12, None)).astype(np.float32)


class MeshRender:
    def __init__(self, camera_distance=1.45, ortho_scale=1.2, scale_factor=1.15):
        self.camera_distance = camera_distance
        self.proj = get_ortho_proj(ortho_scale)
        self.scale_factor = scale_factor

    def load_mesh(self, vertices, faces):
        v = np.asarray(vertices, np.float32).copy()
        f = np.asarray(faces, np.int64)
        v[:, [0, 1]] = -v[:, [0, 1]]      # flip X,Y
        v[:, [1, 2]] = v[:, [2, 1]]        # swap Y,Z
        center = (v.max(0) + v.min(0)) / 2
        scale = np.linalg.norm(v - center, axis=1).max() * 2.0
        v = (v - center) * (self.scale_factor / float(scale))
        self.vtx_pos = v
        self.pos_idx = f
        self.face_world_normals = self._face_normals(v, f)

    @staticmethod
    def _face_normals(v, f):
        tri = v[f[:, :3]]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        return (n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-12, None)).astype(np.float32)

    def _clip(self, elev, azim):
        mv = get_mv_matrix(elev, azim, self.camera_distance)
        posw = np.concatenate([self.vtx_pos, np.ones((self.vtx_pos.shape[0], 1), np.float32)], axis=1)
        pos_cam = posw @ mv.T
        pos_clip = pos_cam @ self.proj.T
        return pos_clip.astype(np.float32)

    def _world_vn(self):
        if getattr(self, "_world_vn_cache", None) is None:
            self._world_vn_cache = mean_vertex_normals(self.vtx_pos.shape[0], self.pos_idx, self.face_world_normals)
        return self._world_vn_cache

    def render_control(self, elev, azim, resolution, bg=(1, 1, 1)):
        """Normal (abs/world) + position control maps in one rasterization (shared geometry)."""
        pos_clip = self._clip(elev, azim)
        findices, bary = _rasterize(pos_clip, self.pos_idx, resolution)
        mask = (findices > 0)[..., None].astype(np.float32)
        bgv = np.array(bg, np.float32)
        normal = cr_raster.interpolate(self._world_vn(), findices, bary, self.pos_idx)
        normal = np.clip(((normal * mask + bgv * (1 - mask)) + 1) * 0.5, 0, 1)
        tex_pos = (0.5 - self.vtx_pos[:, :3] / self.scale_factor).astype(np.float32)
        position = cr_raster.interpolate(tex_pos, findices, bary, self.pos_idx)
        position = np.clip(position * mask + bgv * (1 - mask), 0, 1)
        return normal, position

    def render_normal(self, elev, azim, resolution, bg=(1, 1, 1)):
        pos_clip = self._clip(elev, azim)
        findices, bary = _rasterize(pos_clip, self.pos_idx, resolution)
        normal = cr_raster.interpolate(self._world_vn(), findices, bary, self.pos_idx)
        mask = (findices > 0)[..., None].astype(np.float32)
        normal = normal * mask + np.array(bg, np.float32) * (1 - mask)
        return np.clip((normal + 1) * 0.5, 0, 1)

    def render_position(self, elev, azim, resolution, bg=(1, 1, 1)):
        pos_clip = self._clip(elev, azim)
        findices, bary = _rasterize(pos_clip, self.pos_idx, resolution)
        tex_pos = 0.5 - self.vtx_pos[:, :3] / self.scale_factor
        position = cr_raster.interpolate(tex_pos.astype(np.float32), findices, bary, self.pos_idx)
        mask = (findices > 0)[..., None].astype(np.float32)
        position = position * mask + np.array(bg, np.float32) * (1 - mask)
        return np.clip(position, 0, 1)

    # ---- texture baking (forward-splat 'linear' mode, ports reference MeshRender.back_project) ----

    def set_uv(self, vtx_uv, uv_idx):
        uv = np.asarray(vtx_uv, np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]           # UV-v flip (matches reference set_mesh)
        self.vtx_uv = uv
        self.uv_idx = np.asarray(uv_idx, np.int64)

    def _mv(self, elev, azim):
        mv = get_mv_matrix(elev, azim, self.camera_distance)
        posw = np.concatenate([self.vtx_pos, np.ones((self.vtx_pos.shape[0], 1), np.float32)], axis=1)
        pos_cam = posw @ mv.T
        pos_clip = pos_cam @ self.proj.T
        return pos_cam.astype(np.float32), pos_clip.astype(np.float32)

    def uv_rasterize(self, tex_res):
        """Rasterize the mesh in UV space -> per-texel 3D world position + normal + valid mask."""
        uv = self.vtx_uv
        clip = np.zeros((uv.shape[0], 4), np.float32)
        clip[:, 0] = uv[:, 0] * 2 - 1
        clip[:, 1] = uv[:, 1] * 2 - 1
        clip[:, 3] = 1.0
        fi, ba = _rasterize(clip, self.uv_idx, tex_res)
        tex_pos = cr_raster.interpolate(self.vtx_pos, fi, ba, self.uv_idx)
        vn = mean_vertex_normals(self.vtx_pos.shape[0], self.pos_idx,
                                 self._face_normals(self.vtx_pos, self.pos_idx))
        tex_nrm = cr_raster.interpolate(vn, fi, ba, self.uv_idx)
        return tex_pos, tex_nrm, (fi > 0)

    @staticmethod
    def _bilinear(img, row_f, col_f):
        H, W = img.shape[:2]
        r0 = np.clip(np.floor(row_f).astype(int), 0, H - 2); c0 = np.clip(np.floor(col_f).astype(int), 0, W - 2)
        fr = (row_f - r0)[:, None]; fc = (col_f - c0)[:, None]
        return (img[r0, c0] * (1 - fr) * (1 - fc) + img[r0, c0 + 1] * (1 - fr) * fc
                + img[r0 + 1, c0] * fr * (1 - fc) + img[r0 + 1, c0 + 1] * fr * fc)

    def bake_multi(self, view_sets, elevs, azims, texture_size=1024, exp=6.0, weights=None, eps=0.05):
        """Back-sample bake for several color sets (e.g. albedo + metallic-roughness) sharing one
        geometry pass: UV rasterization, per-view depth/occlusion/weights computed once; only the
        per-set bilinear color gather differs. Returns (list of textures, covered mask)."""
        weights = weights or [1.0] * len(elevs)
        tex_pos, tex_nrm, mask = self.uv_rasterize(texture_size)
        idx = np.argwhere(mask)                              # [K,2] (row,col)
        P = tex_pos[mask]; Nn = tex_nrm[mask]
        Pw = np.concatenate([P, np.ones((P.shape[0], 1), np.float32)], 1)
        nsets = len(view_sets)
        accs = [np.zeros((P.shape[0], 3), np.float64) for _ in range(nsets)]
        wsum = np.zeros((P.shape[0], 1), np.float64)
        cos_thr = math.cos(math.radians(75.0))
        posw = np.concatenate([self.vtx_pos, np.ones((self.vtx_pos.shape[0], 1), np.float32)], 1)
        for vi in range(len(elevs)):
            e, a, w = elevs[vi], azims[vi], weights[vi]
            H, Wd = view_sets[0][vi].shape[:2]
            mv = get_mv_matrix(e, a, self.camera_distance)
            pos_cam_all = posw @ mv.T
            pos_clip_all = (pos_cam_all @ self.proj.T).astype(np.float32)
            fi_d, ba_d = _rasterize(pos_clip_all, self.pos_idx, (H, Wd))
            depth_map = cr_raster.interpolate(pos_cam_all[:, 2:3].astype(np.float32), fi_d, ba_d, self.pos_idx)[..., 0]
            covd = fi_d > 0
            pc = Pw @ mv.T
            ndc = (pc @ self.proj.T)[:, :2] / (pc @ self.proj.T)[:, 3:4]
            zc = pc[:, 2]
            col_f = (ndc[:, 0] * 0.5 + 0.5) * (Wd - 1) + 0.5
            row_f = (0.5 + 0.5 * ndc[:, 1]) * (H - 1) + 0.5
            inside = (col_f >= 0) & (col_f <= Wd - 1) & (row_f >= 0) & (row_f <= H - 1)
            ri = np.clip(row_f.astype(int), 0, H - 1); ci = np.clip(col_f.astype(int), 0, Wd - 1)
            vis = inside & covd[ri, ci] & (np.abs(zc - depth_map[ri, ci]) < eps)
            cam_n = Nn @ mv[:3, :3].T
            cos = -cam_n[:, 2] / np.clip(np.linalg.norm(cam_n, axis=1), 1e-8, None)
            cosw = np.where(cos >= cos_thr, np.clip(cos, 0, None) ** exp, 0.0) * w
            wgt = (vis * cosw)[:, None]; wsum += wgt
            rfc, cfc = np.clip(row_f, 0, H - 1), np.clip(col_f, 0, Wd - 1)
            for si in range(nsets):
                accs[si] += self._bilinear(np.asarray(view_sets[si][vi], np.float32), rfc, cfc) * wgt
        covered = np.zeros((texture_size, texture_size), bool)
        ok = wsum[:, 0] > 1e-8
        covered[idx[ok, 0], idx[ok, 1]] = True
        texs = []
        for si in range(nsets):
            tex = np.zeros((texture_size, texture_size, 3), np.float32)
            tex[idx[:, 0], idx[:, 1]] = (accs[si] / np.clip(wsum, 1e-8, None)).astype(np.float32)
            texs.append(tex)
        return texs, covered

    def bake(self, views, elevs, azims, texture_size=1024, exp=6.0, weights=None, eps=0.05, bake_render=None):
        """Back-sample (gather) bake of a single color set. See bake_multi."""
        texs, covered = self.bake_multi([views], elevs, azims, texture_size, exp, weights, eps)
        return texs[0], covered

    @staticmethod
    def inpaint(texture, mask):
        """Fill un-painted texels: nearest-EDT (complete) then Navier-Stokes smoothing on the holes
        (mirrors the reference meshVerticeInpaint + cv2.inpaint final step)."""
        from scipy import ndimage
        out = np.clip(texture, 0, 1).astype(np.float32)
        holes = ~mask
        if holes.any() and mask.any():
            idx = ndimage.distance_transform_edt(holes, return_distances=False, return_indices=True)
            out = out[tuple(idx)]
            try:
                import cv2
                m8 = (holes.astype(np.uint8)) * 255
                sm = cv2.inpaint((out * 255).astype(np.uint8), m8, 3, cv2.INPAINT_NS)
                out = sm.astype(np.float32) / 255.0
            except Exception:
                pass
        return np.clip(out, 0, 1)
