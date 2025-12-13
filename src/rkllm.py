import multiprocessing
import base64
import ctypes
import textwrap
from abc import ABC, abstractmethod
from enum import Enum
import cv2
import torch
import numpy as np
from .callback import (
    callback_impl, gui_actor_callback_impl,
    embed_callback_impl, rerank_callback_impl
)
from .classes import *
from .img_encoder import ImageEncoder
from .ztu_somemodelruntime_rknnlite2 import SessionOptions, InferenceSession

# Connect the callback function between Python and C++
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)
gui_actor_callback = callback_type(gui_actor_callback_impl)
embed_callback = callback_type(embed_callback_impl)
rerank_callback = callback_type(rerank_callback_impl)


class RKModelType(Enum):
    LANGUAGE = "language"
    EMBED = "embed"
    RERANKER = "reranker"


class RKModel(ABC):
    def __init__(self, model_path, model_dir, temperature=0.8, context_length=4096, **kwargs):
        self.format_schema = None
        self.format_type = None
        self.format_options = {}
        self.model_dir = model_dir

        self.handle = RKLLM_Handle_t()

        self._setup_functions()

    def _setup_functions(self):
        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int

        self.rkllm_createDefaultParam = rkllm_lib.rkllm_createDefaultParam
        self.rkllm_createDefaultParam.argtypes = []
        self.rkllm_createDefaultParam.restype = RKLLMParam

        self.rkllm_clear_kv_cache = rkllm_lib.rkllm_clear_kv_cache
        self.rkllm_clear_kv_cache.argtypes = [RKLLM_Handle_t, ctypes.c_int]
        self.rkllm_clear_kv_cache.restype = ctypes.c_int

        self.rkllm_set_chat_template = rkllm_lib.rkllm_set_chat_template
        self.rkllm_set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.rkllm_set_chat_template.restype = ctypes.c_int

    @abstractmethod
    def get_model_type(self) -> RKModelType:
        pass

    @abstractmethod
    def run(self, prompt, **kwargs):
        pass

    def release(self):
        if self.handle and self.handle.value:
            self.rkllm_destroy(self.handle)
            self.handle = None


# Define the RKLLM class, which includes initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM(RKModel):
    def __init__(self, model_path, model_dir, temperature=0.8, context_length=4096, **kwargs):
        super(RKLLM, self).__init__(model_path, model_dir, temperature, context_length, **kwargs)

        rkllm_param = self.rkllm_createDefaultParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        rkllm_param.max_context_len = context_length
        rkllm_param.max_new_tokens = context_length
        rkllm_param.skip_special_token = kwargs.get("skip_special_token", True)
        rkllm_param.n_keep = -1
        rkllm_param.top_k = 1
        rkllm_param.top_p = 0.9
        rkllm_param.temperature = temperature
        rkllm_param.repeat_penalty = 1.1
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0

        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1

        rkllm_param.is_async = False

        rkllm_param.img_start = "<|vision_start|>".encode('utf-8')
        rkllm_param.img_end = "<|vision_end|>".encode('utf-8')
        rkllm_param.img_content = "<|image_pad|>".encode('utf-8')

        rkllm_param.extend_param.base_domain_id = kwargs.get("base_domain_id", 0)
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = 0
        rkllm_param.extend_param.enabled_cpus_num = multiprocessing.cpu_count()
        rkllm_param.extend_param.enabled_cpus_mask = (1<<(rkllm_param.extend_param.enabled_cpus_num+1))-1

        self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), kwargs.get("callback", callback))

        self.lora_adapter_path = None
        self.lora_model_name = None
        lora_model_path = kwargs.get("lora_model_path")
        if lora_model_path:
            self.lora_adapter_path = lora_model_path
            self.lora_adapter_name = "test"

            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p((self.lora_adapter_path).encode('utf-8'))
            lora_adapter.lora_adapter_name = ctypes.c_char_p((self.lora_adapter_name).encode('utf-8'))
            lora_adapter.scale = 1.0

            rkllm_load_lora = rkllm_lib.rkllm_load_lora
            rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            rkllm_load_lora.restype = ctypes.c_int
            rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))

        self.prompt_cache_path = None
        prompt_cache_path = kwargs.get("prompt_cache_path")
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path

            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode('utf-8')))

    def get_model_type(self):
        return RKModelType.LANGUAGE

    def run(self, prompt, **kwargs):
        rkllm_lora_params = None
        if self.lora_model_name:
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((self.lora_model_name).encode('utf-8'))

        rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        rkllm_infer_params.lora_params = ctypes.byref(rkllm_lora_params) if rkllm_lora_params else None
        rkllm_infer_params.keep_history = 0

        rkllm_input = RKLLMInput()
        rkllm_input.role = b"user"
        rkllm_input.enable_thinking = False
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode("utf-8"))

        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)


class RKLLM_IMG(RKLLM):
    def __init__(self, model_path, model_dir, temperature=0.8, context_length=2048, **kwargs):
        super(RKLLM_IMG, self).__init__(model_path, model_dir, temperature, context_length, **kwargs)

        image_emb_model_path = kwargs.get("image_emb_model_path", None)
        if image_emb_model_path:
            self.img_encoder = ImageEncoder(image_emb_model_path)
        else:
            self.img_encoder = None

        # TODO: make these configuable?
        # See bin/img_encoder.cpp
        self.image_width = 392
        self.image_height = 392
        self.n_image_tokens = 196

    def run(self, prompt, **kwargs):
        images = kwargs.get("images", [])
        if not (images and self.img_encoder):
            super(RKLLM_IMG, self).run(prompt, **kwargs)
            return

        image_embed = self.img_encoder.encode_image(images[0])

        rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        rkllm_infer_params.lora_params = None
        rkllm_infer_params.keep_history = 0

        rkllm_input = RKLLMInput()
        rkllm_input.role = b"user"
        rkllm_input.enable_thinking = False
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_MULTIMODAL
        rkllm_input.input_data.multimodal_input.prompt = ctypes.c_char_p(prompt.encode("utf-8"))
        rkllm_input.input_data.multimodal_input.image_embed = image_embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rkllm_input.input_data.multimodal_input.n_image_tokens = self.n_image_tokens
        rkllm_input.input_data.multimodal_input.n_image = 1
        rkllm_input.input_data.multimodal_input.image_height = self.image_height
        rkllm_input.input_data.multimodal_input.image_width = self.image_width

        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)

    def release(self):
        super(RKLLM_IMG, self).release()
        if self.img_encoder:
            self.img_encoder.release()
            self.img_encoder = None


class RKLLM_GUI_ACTOR(RKLLM):
    def __init__(self, model_path, model_dir, temperature=0.8, context_length=2048, **kwargs):
        kwargs["skip_special_token"] = False
        kwargs["base_domain_id"] = 1
        kwargs["callback"] = gui_actor_callback
        super(RKLLM_GUI_ACTOR, self).__init__(model_path, model_dir, temperature, context_length, **kwargs)

        vision_encoder_model_path = kwargs.get("vision_encoder_model_path", None)
        pointer_head_model_path = kwargs.get("pointer_head_model_path", None)
        if vision_encoder_model_path and pointer_head_model_path:
            session_options = SessionOptions()
            session_options.intra_op_nu_threads = 3

            self.vision_encoder = InferenceSession(vision_encoder_model_path, session_options)
            self.vision_encoder_input_name = self.vision_encoder.get_inputs()[0].name
            self.vision_encoder_output_name = self.vision_encoder.get_outputs()[0].name

            self.pointer_head = InferenceSession(pointer_head_model_path, session_options)
            self.pointer_head_input_names = [i.name for i in self.pointer_head.get_inputs()]
            self.pointer_head_output_names = [o.name for o in self.pointer_head.get_outputs()]
        else:
            self.vision_encoder = None
            self.pointer_head = None

        self.image_width = 1148
        self.image_height = 896

        self.rkllm_set_chat_template(
            self.handle,
            ctypes.c_char_p(b""),  # system prompt
            ctypes.c_char_p(b""),  # prompt prefix
            ctypes.c_char_p(b""),  # prompt postfix
        )

    def run(self, prompt, **kwargs):
        image = kwargs.get("image", None)
        if not (image and self.vision_encoder and self.pointer_head):
            super(RKLLM_GUI_ACTOR, self).run(prompt, **kwargs)
            return

        img_vec_output = self.get_img_vec(image)
        self.img_vec_output = img_vec_output  # save for pointer head
        print(f"Vision encoder output shape: {img_vec_output.shape}")
        img_vec = img_vec_output.flatten().astype(np.float32)
        n_image_tokens = img_vec_output.shape[0]
        gui_actor_prompt = self.get_gui_actor_prompt(prompt)

        rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER
        rkllm_infer_params.keep_history = 0

        rkllm_input = RKLLMInput()
        rkllm_input.role = b"user"
        rkllm_input.enable_thinking = False
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_MULTIMODAL
        rkllm_input.input_data.multimodal_input.prompt = gui_actor_prompt.encode("utf-8")
        rkllm_input.input_data.multimodal_input.image_embed = img_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rkllm_input.input_data.multimodal_input.n_image_tokens = n_image_tokens
        rkllm_input.input_data.multimodal_input.n_image = 1
        rkllm_input.input_data.multimodal_input.image_height = self.image_height
        rkllm_input.input_data.multimodal_input.image_width = self.image_width

        self.rkllm_run(
            self.handle, ctypes.byref(rkllm_input),
            ctypes.byref(rkllm_infer_params), None
        )

    def run_pointer_head(self, hidden_states, input_image_data, label=False):
        if self.img_vec_output is None or self.pointer_head is None:
            return None, 0, 0

        decoder_hidden_states = hidden_states[-2].reshape(1, -1).astype(np.float32)
        image_embeds = self.img_vec_output.astype(np.float32)
        print(f"Pointer head inputs shapes: enc={image_embeds.shape}, dec={decoder_hidden_states.shape}")
        attn_scores_np = self.pointer_head.run(
            self.pointer_head_output_names,
            {
                self.pointer_head_input_names[0]: image_embeds,
                self.pointer_head_input_names[1]: decoder_hidden_states
            }
        )[0]
        attn_scores = torch.from_numpy(attn_scores_np)
        print(f"Got attention scores, shape: {attn_scores.shape}")

        n_height = self.image_height / 28
        n_width = self.image_width / 28
        best_point, _, _, _ = self.get_prediction_region_point(
            attn_scores, n_width, n_height, return_all_regions=True
        )
        print(f"Predicted Best Point (normalized): x={best_point[0]:.4f}, y={best_point[1]:.4f}")

        img = self.base64_to_cv2(input_image_data)
        h, w, _ = img.shape

        pred_x_padded = best_point[0] * self.image_width
        pred_y_padded = best_point[1] * self.image_height

        top = int(round(self.dh - 0.1))
        left = int(round(self.dw - 0.1))

        px = int((pred_x_padded - left) / self.ratio)
        py = int((pred_y_padded - top) / self.ratio)

        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)

        if label:
            cv2.drawMarker(img, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 25, 1)
            return self.cv2_to_base64(img), px, py
        else:
            return None, px, py

    def get_gui_actor_prompt(self, task):
        return f"""<|im_start|>system
You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>).<|im_end|>
<|im_start|>user
<image>{task}<|im_end|>
<|im_start|>assistant<|recipient|>os
pyautogui.click(<|pointer_start|><|pointer_pad|>"""

    def get_img_vec(self, image_data):
        img = self.base64_to_cv2(image_data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img, ratio, (dw, dh) = self.letterbox(img, (self.image_height, self.image_width))
        print(f"resized_img shape: {resized_img.shape}, ratio: {ratio}, (dw, dh): {dw}, {dh}")
        self.ratio = ratio
        self.dw = dw
        self.dh = dh
        # Normalize and prepare for ONNX model
        input_tensor = resized_img.astype(np.float32)
        # Normalize using preprocessor config values
        input_tensor = (input_tensor / 255.0 - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
        # Convert to NCHW format
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
        # Add batch dimension -> (1, 3, 392, 392)
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

        return self.vision_encoder.run(
            [self.vision_encoder_output_name],
            {self.vision_encoder_input_name: input_tensor.astype(np.float32)}
        )[0]

    def letterbox(self, img, new_shape, color=(127.5, 127.5, 127.5)):
        """Resize and pad image while preserving aspect ratio. """
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, r, (dw, dh)

    def get_prediction_region_point(self, attn_scores, n_width, n_height, top_n=30, activation_threshold=0.3, return_all_regions=True, rect_center=False):
        # Get the highest activation value and threshold
        max_score = attn_scores[0].max().item()
        threshold = max_score * activation_threshold
        # Select all patches above the threshold
        mask = attn_scores[0] > threshold
        valid_indices = torch.nonzero(mask).squeeze(-1)
        topk_values = attn_scores[0][valid_indices]
        topk_indices = valid_indices

        # Convert indices to 2D coordinates
        topk_coords = []
        for idx in topk_indices.tolist():
            y = idx // n_width
            x = idx % n_width
            topk_coords.append((y, x, idx))

        # Divide into connected regions
        regions = []
        visited = set()
        for i, (y, x, idx) in enumerate(topk_coords):
            if idx in visited:
                continue

            # Start a new region
            region = [(y, x, idx, topk_values[i].item())]
            visited.add(idx)
            queue = [(y, x, idx, topk_values[i].item())]

            # BFS to find connected points
            while queue:
                cy, cx, c_idx, c_val = queue.pop(0)

                # Check 4 adjacent directions
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx

                    # Check if this adjacent point is in the topk list
                    for j, (ty, tx, t_idx) in enumerate(topk_coords):
                        if ty == ny and tx == nx and t_idx not in visited:
                            visited.add(t_idx)
                            region.append((ny, nx, t_idx, topk_values[j].item()))
                            queue.append((ny, nx, t_idx, topk_values[j].item()))

            regions.append(region)

        # Calculate the average activation value for each region
        region_scores = []
        region_centers = []
        region_points = []

        for region in regions:
            # Calculate average score for the region
            avg_score = sum(item[3] for item in region) / len(region)
            region_scores.append(avg_score)

            # Calculate normalized center coordinates for each patch, then take the average
            normalized_centers = []
            weights = []
            y_coords = set()
            x_coords = set()

            for y, x, _, score in region:
                # Normalized coordinates of the center point for each patch
                center_y = (y + 0.5) / n_height
                center_x = (x + 0.5) / n_width
                normalized_centers.append((center_x, center_y))
                weights.append(score)

                y_coords.add(center_y)
                x_coords.add(center_x)

            region_points.append(normalized_centers)

            # Calculate the average of normalized coordinates as the region center
            if not rect_center:
                # Weighted average
                total_weight = sum(weights)
                weighted_x = sum(nc[0] * w for nc, w in zip(normalized_centers, weights)) / total_weight
                weighted_y = sum(nc[1] * w for nc, w in zip(normalized_centers, weights)) / total_weight
                avg_center_x, avg_center_y = weighted_x, weighted_y
                # # Simple average
                # avg_center_x = sum(nc[0] for nc in normalized_centers) / len(normalized_centers)
                # avg_center_y = sum(nc[1] for nc in normalized_centers) / len(normalized_centers)
            else:
                avg_center_x = sum(x_coords) / len(x_coords)
                avg_center_y = sum(y_coords) / len(y_coords)
            region_centers.append((avg_center_x, avg_center_y))

        # Select the region with the highest average activation value
        sorted_indices = sorted(range(len(region_scores)), key=lambda i: region_scores[i], reverse=True)
        sorted_scores = [region_scores[i] for i in sorted_indices]
        sorted_centers = [region_centers[i] for i in sorted_indices]
        sorted_points = [region_points[i] for i in sorted_indices]
        best_point = sorted_centers[0]

        if return_all_regions:
            # Outputs:
            # 1. best_point: the center point of the region with the highest average activation value
            # 2. sorted_centers: the center points of all regions, sorted by the average activation value in descending order
            # 3. sorted_scores: the average activation values of all regions, sorted in descending order
            # 4. sorted_points: the normalized center coordinates of all patches, sorted by the average activation value in descending order
            return best_point, sorted_centers, sorted_scores, sorted_points
        else:
            return best_point

    def base64_to_cv2(self, image_data):
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def cv2_to_base64(self, image):
        _, im_arr = cv2.imencode(".jpg", image)
        im_bytes = im_arr.tobytes()
        return base64.b64encode(im_bytes).decode("utf-8")

    def release(self):
        super(RKLLM_GUI_ACTOR, self).release()
        if self.vision_encoder is not None:
            self.vision_encoder.close()
            self.vision_encoder = None
        if self.pointer_head is not None:
            self.pointer_head.close()
            self.pointer_head = None


class RKEMBED(RKModel):
    def __init__(self, model_path, model_dir, temperature=1.0, context_length=2048, **kwargs):
        super(RKEMBED, self).__init__(model_path, model_dir, temperature, context_length, **kwargs)

        rkllm_param = self.rkllm_createDefaultParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        rkllm_param.max_context_len = context_length
        rkllm_param.max_new_tokens = 1
        rkllm_param.top_k = 1
        rkllm_param.top_p = 1.0
        rkllm_param.temperature = 1.0

        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.embed_flash = 0
        rkllm_param.extend_param.enabled_cpus_num = multiprocessing.cpu_count()
        rkllm_param.extend_param.enabled_cpus_mask = (1<<(rkllm_param.extend_param.enabled_cpus_num+1))-1

        self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), embed_callback)

        self.rkllm_set_chat_template(
            self.handle,
            ctypes.c_char_p(b""),  # system prompt
            ctypes.c_char_p(b""),  # prompt prefix
            ctypes.c_char_p(b""),  # prompt postfix
        )

    def get_model_type(self):
        return RKModelType.EMBED

    def run(self, prompt, **kwargs):
        _ = self.rkllm_clear_kv_cache(self.handle, ctypes.c_int(0))

        rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER
        rkllm_infer_params.keep_history = 0

        rkllm_input = RKLLMInput()
        rkllm_input.role = b"user"
        rkllm_input.enable_thinking = False
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode("utf-8"))

        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)


class RKRERANKER(RKModel):
    def __init__(self, model_path, model_dir, temperature=1.0, context_length=2048, **kwargs):
        super(RKRERANKER, self).__init__(model_path, model_dir, temperature, context_length, **kwargs)

        rkllm_param = self.rkllm_createDefaultParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        rkllm_param.max_context_len = context_length
        rkllm_param.max_new_tokens = 1
        rkllm_param.top_k = 1
        rkllm_param.top_p = 1.0
        rkllm_param.temperature = 0.0

        rkllm_param.extend_param.base_domain_id = 1
        rkllm_param.extend_param.embed_flash = 0
        rkllm_param.extend_param.enabled_cpus_num = multiprocessing.cpu_count()
        rkllm_param.extend_param.enabled_cpus_mask = (1<<(rkllm_param.extend_param.enabled_cpus_num+1))-1

        self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), rerank_callback)

        self.rkllm_set_chat_template(
            self.handle,
            ctypes.c_char_p(b""),  # system prompt
            ctypes.c_char_p(b""),  # prompt prefix
            ctypes.c_char_p(b""),  # prompt postfix
        )

    def get_model_type(self):
        return RKModelType.RERANKER

    def format_rerank_input(self, prompt, document, instruction):
        if not instruction:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'

        formatted_input = textwrap.dedent(f"""<Instruct>: {instruction}
        <Query>: {prompt}
        <Document>: {document}""")
        return formatted_input

    def run(self, prompt, **kwargs):
        _ = self.rkllm_clear_kv_cache(self.handle, ctypes.c_int(0))

        document = kwargs.get("document", "")
        instruction = kwargs.get("instruction", None)
        rerank_input = self.format_rerank_input(prompt, document, instruction)

        rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GET_LOGITS
        rkllm_infer_params.keep_history = 0

        rkllm_input = RKLLMInput()
        rkllm_input.role = b"user"
        rkllm_input.enable_thinking = False
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(rerank_input.encode("utf-8"))

        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)


class RKModelFactory:
    @classmethod
    def create_model(cls, model_type, model_path, model_dir, temperature=0.8, context_length=2048, **kwargs) -> RKModel:
        model_cls = None
        if model_type == RKModelType.LANGUAGE:
            if kwargs.get("image_emb_model_path", None):
                model_cls = RKLLM_IMG
            elif kwargs.get("vision_encoder_model_path", None) and kwargs.get("pointer_head_model_path", None):
                model_cls = RKLLM_GUI_ACTOR
            else:
                model_cls = RKLLM
        elif model_type == RKModelType.EMBED:
            model_cls = RKEMBED
        elif model_type == RKModelType.RERANKER:
            model_cls = RKRERANKER

        if not model_cls:
            return None

        return model_cls(model_path, model_dir, temperature, context_length, **kwargs)
