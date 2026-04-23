# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


import logging
import numpy as np
import os
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from typing import List, Union, Optional


from .differentiable_renderer.mesh_render import MeshRender
from .utils.dehighlight_utils import Light_Shadow_Remover
from .utils.multiview_utils import Multiview_Diffusion_Net
from .utils.imagesuper_utils import Image_Super_Net
from .utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


class Hunyuan3DTexGenConfig:

    def __init__(
        self,
        light_remover_ckpt_path,
        multiview_ckpt_path,
        subfolder_name,
        diffusion_backend='pytorch',
        mlx_weights_path=None,
        mlx_profile=None,
    ):
        # Prefer CUDA, then MPS, then CPU.
        self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path

        self.diffusion_backend = diffusion_backend
        self.mlx_weights_path = mlx_weights_path
        self.mlx_profile = mlx_profile

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        self.render_size = 2048
        self.texture_size = 2048
        self.bake_exp = 4
        self.merge_method = 'fast'

        self.pipe_dict = {
            'hunyuan3d-paint-v2-0': 'hunyuanpaint',
            'hunyuan3d-paint-v2-0-turbo': 'hunyuanpaint-turbo',
            'hunyuan3d-paint-v2-1': 'hunyuanpaintpbr',
            'hunyuan3d-paintpbr-v2-1': 'hunyuanpaintpbr',
            'hunyuan3d-paint-v2-1-turbo': 'hunyuanpaintpbr',
        }
        self.pipe_name = self.pipe_dict.get(subfolder_name, 'hunyuanpaint-turbo' if 'turbo' in subfolder_name else 'hunyuanpaint')


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(
        cls,
        model_path,
        subfolder='hunyuan3d-paint-v2-0-turbo',
        diffusion_backend='pytorch',
        mlx_weights_path=None,
    ):
        def _infer_mlx_profile(subfolder_name: str) -> str:
            if 'paintpbr' in subfolder_name or 'v2-1' in subfolder_name:
                return 'paint-pbr-2.1'
            return 'paint-2.0'

        def _mlx_hf_source(profile: str, subfolder_name: str):
            if profile == 'paint-2.0':
                repo_id = 'zimengxiong/Hunyuan3D-2.0-Paint-MLX'
                subdir = '2.0-turbo' if 'turbo' in subfolder_name else '2.0'
            else:
                repo_id = 'zimengxiong/Hunyuan3D-2.1-Paint-MLX'
                subdir = None
            return repo_id, subdir

        def _default_mlx_weights_path(multiview_model_path: str, profile: str, subfolder_name: str):
            parent = os.path.dirname(multiview_model_path)
            if profile == 'paint-2.0':
                local_subdir = '2.0-turbo' if 'turbo' in subfolder_name else '2.0'
                candidates = [
                    os.path.join(parent, 'hunyuan3d-2.0-mlx', local_subdir),
                    os.path.join(multiview_model_path, 'mlx_weights'),
                ]
            else:
                candidates = [
                    os.path.join(parent, 'hunyuan3d-2.1-mlx'),
                    os.path.join(multiview_model_path, 'mlx_weights'),
                ]
            for c in candidates:
                if os.path.exists(os.path.join(c, 'unet.npz')):
                    return c
            return candidates[0]

        def _download_mlx_weights_from_hf(profile: str, subfolder_name: str) -> str:
            import huggingface_hub

            repo_id, subdir = _mlx_hf_source(profile, subfolder_name)
            if subdir:
                snapshot = huggingface_hub.snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[f'{subdir}/*'],
                )
                return os.path.join(snapshot, subdir)

            snapshot = huggingface_hub.snapshot_download(
                repo_id=repo_id,
                allow_patterns=['*.npz', 'README.md'],
            )
            return snapshot

        original_model_path = model_path
        delight_model_path = None
        if not os.path.exists(model_path):
            # try local path
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, model_path))

            multiview_model_path = os.path.join(model_path, subfolder)
            candidate_delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            if os.path.exists(candidate_delight_model_path):
                delight_model_path = candidate_delight_model_path

            if not os.path.exists(multiview_model_path):
                try:
                    import huggingface_hub
                    model_path = huggingface_hub.snapshot_download(
                        repo_id=original_model_path, allow_patterns=[f'{subfolder}/*']
                    )
                    multiview_model_path = os.path.join(model_path, subfolder)
                    candidate_delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
                    if os.path.exists(candidate_delight_model_path):
                        delight_model_path = candidate_delight_model_path
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"Something wrong while loading {model_path}")
        else:
            candidate_delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            if os.path.exists(candidate_delight_model_path):
                delight_model_path = candidate_delight_model_path
            multiview_model_path = os.path.join(model_path, subfolder)

        mlx_profile = _infer_mlx_profile(subfolder)
        resolved_mlx_weights_path = mlx_weights_path
        if diffusion_backend == 'mlx' and resolved_mlx_weights_path is None:
            resolved_mlx_weights_path = _default_mlx_weights_path(multiview_model_path, mlx_profile, subfolder)
            if not os.path.exists(os.path.join(resolved_mlx_weights_path, 'unet.npz')):
                resolved_mlx_weights_path = _download_mlx_weights_from_hf(mlx_profile, subfolder)

        return cls(
            Hunyuan3DTexGenConfig(
                delight_model_path,
                multiview_model_path,
                subfolder,
                diffusion_backend=diffusion_backend,
                mlx_weights_path=resolved_mlx_weights_path,
                mlx_profile=mlx_profile,
            )
        )
            
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            device=self.config.device,
            raster_mode='auto')

        self.load_models()

    def load_models(self):
        # Empty CUDA cache only when CUDA is active.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Load model
        self.models['delight_model'] = None
        if self.config.light_remover_ckpt_path and os.path.exists(self.config.light_remover_ckpt_path):
            self.models['delight_model'] = Light_Shadow_Remover(self.config)
        self.models['multiview_model'] = Multiview_Diffusion_Net(self.config)
        # self.models['super_model'] = Image_Super_Net(self.config)

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        if self.models['delight_model'] is not None:
            self.models['delight_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        self.models['multiview_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render.render_normal(
                elev, azim, use_abs_coor=use_abs_coor, return_type='pl')
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(
                elev, azim, return_type='pl')
            position_maps.append(position_map)

        return position_maps

    def bake_from_multiview(self, views, camera_elevs,
                            camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim)
            project_cos_map = weight * (project_cos_map ** self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)

        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8

    def texture_inpaint(self, texture, mask):

        texture_np = self.render.uv_inpaint(texture, mask)
        texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture

    def recenter_image(self, image, border_ratio=0.2):
        if image.mode == 'RGB':
            return image
        elif image.mode == 'L':
            image = image.convert('RGB')
            return image

        alpha_channel = np.array(image)[:, :, 3]
        non_zero_indices = np.argwhere(alpha_channel > 0)
        if non_zero_indices.size == 0:
            raise ValueError("Image is fully transparent")

        min_row, min_col = non_zero_indices.min(axis=0)
        max_row, max_col = non_zero_indices.max(axis=0)

        cropped_image = image.crop((min_col, min_row, max_col + 1, max_row + 1))

        width, height = cropped_image.size
        border_width = int(width * border_ratio)
        border_height = int(height * border_ratio)

        new_width = width + 2 * border_width
        new_height = height + 2 * border_height

        square_size = max(new_width, new_height)

        new_image = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))

        paste_x = (square_size - new_width) // 2 + border_width
        paste_y = (square_size - new_height) // 2 + border_height

        new_image.paste(cropped_image, (paste_x, paste_y))
        return new_image

    @torch.no_grad()
    def __call__(self, mesh, image):

        if not isinstance(image, List):
            image = [image]

        images_prompt = []
        for i in range(len(image)):
            if isinstance(image[i], str):
                image_prompt = Image.open(image[i])
            else:
                image_prompt = image[i]
            images_prompt.append(image_prompt)
            
        images_prompt = [self.recenter_image(image_prompt) for image_prompt in images_prompt]

        # On MPS, the delight preprocessor can produce unstable outputs (noise).
        # Skip it unless explicitly enabled and available.
        use_delight = (
            os.environ.get('HY3D_USE_DELIGHT', '0') == '1'
            and self.config.device != 'mps'
            and self.models['delight_model'] is not None
        )
        if use_delight:
            images_prompt = [self.models['delight_model'](image_prompt) for image_prompt in images_prompt]

        overlap_uv_unwrap = os.environ.get('HY3D_OVERLAP_UV_UNWRAP', '0') == '1'
        if overlap_uv_unwrap:
            # Fast path: overlap UV unwrapping on CPU with multiview diffusion on GPU.
            # This improves throughput substantially, but can introduce small texture
            # differences because control views are rendered before UV seam splits.
            raw_mesh = mesh.copy() if hasattr(mesh, 'copy') else mesh
            self.render.load_mesh(raw_mesh)
        else:
            mesh = mesh_uv_wrap(mesh)
            self.render.load_mesh(mesh)

        selected_camera_elevs, selected_camera_azims, selected_view_weights = \
            self.config.candidate_camera_elevs, self.config.candidate_camera_azims, self.config.candidate_view_weights

        normal_maps = self.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
        position_maps = self.render_position_multiview(
            selected_camera_elevs, selected_camera_azims)

        uv_future = None
        mesh_executor = None
        uv_ready_mesh = mesh
        if overlap_uv_unwrap:
            if hasattr(raw_mesh.visual, 'uv') and raw_mesh.visual.uv is not None:
                uv_ready_mesh = raw_mesh
            else:
                mesh_executor = ThreadPoolExecutor(max_workers=1)
                uv_future = mesh_executor.submit(mesh_uv_wrap, mesh.copy() if hasattr(mesh, 'copy') else mesh)
                uv_ready_mesh = None

        camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[
            elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                       zip(selected_camera_azims, selected_camera_elevs)]
        multiviews = self.models['multiview_model'](images_prompt, normal_maps + position_maps, camera_info)

        if uv_future is not None:
            uv_ready_mesh = uv_future.result()
            mesh_executor.shutdown(wait=False)
            self.render.load_mesh(uv_ready_mesh)

        for i in range(len(multiviews)):
            # multiviews[i] = self.models['super_model'](multiviews[i])
            multiviews[i] = multiviews[i].resize(
                (self.config.render_size, self.config.render_size))

        texture, mask = self.bake_from_multiview(multiviews,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=self.config.merge_method)

        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        texture = self.texture_inpaint(texture, mask_np)

        self.render.set_texture(texture)
        textured_mesh = self.render.save_mesh()

        return textured_mesh
