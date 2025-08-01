import multiprocessing
import ctypes
import textwrap
from abc import ABC, abstractmethod
from enum import Enum
from .callback import callback_impl, embed_callback_impl, rerank_callback_impl
from .classes import *

# Connect the callback function between Python and C++
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)
embed_callback = callback_type(embed_callback_impl)
rerank_callback = callback_type(rerank_callback_impl)


class RKModelType(Enum):
    LANGUAGE = "language"
    EMBED = "embed"
    RERANKER = "reranker"


class RKModel(ABC):
    def __init__(self, model_path, model_dir, temperature=0.8, context_length=2048, **kwargs):
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
    def __init__(self, model_path, model_dir, temperature=0.8, context_length=2048, **kwargs):
        super(RKLLM, self).__init__(model_path, model_dir, temperature, context_length, **kwargs)

        rkllm_param = self.rkllm_createDefaultParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        rkllm_param.max_context_len = context_length
        rkllm_param.max_new_tokens = -1
        rkllm_param.skip_special_token = True

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

        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.enabled_cpus_num = multiprocessing.cpu_count()
        rkllm_param.extend_param.enabled_cpus_mask = (1<<(rkllm_param.extend_param.enabled_cpus_num+1))-1

        self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)

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

        rkllm_input = RKLLMInput()
        image_embed = kwargs.get("img_emb")
        if image_embed is None:
            rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
            rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode("utf-8"))
        else:
            rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_MULTIMODAL
            rkllm_input.input_data.multimodal_input.prompt = ctypes.c_char_p(prompt.encode("utf-8"))
            rkllm_input.input_data.multimodal_input.image_embed = image_embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            # TODO: make these configuable?
            # See bin/img_encoder.cpp
            rkllm_input.input_data.multimodal_input.n_image_tokens = 196
            rkllm_input.input_data.multimodal_input.n_image = 1
            rkllm_input.input_data.multimodal_input.image_height = 392
            rkllm_input.input_data.multimodal_input.image_with = 392

        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)


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

    def get_model_type(self):
        return RKModelType.EMBED

    def run(self, prompt, **kwargs):
        _ = self.rkllm_clear_kv_cache(self.handle, ctypes.c_int(0))

        rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER
        rkllm_infer_params.keep_history = 0

        rkllm_input = RKLLMInput()
        rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
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
        rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(rerank_input.encode("utf-8"))

        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)


class RKModelFactory:
    @classmethod
    def create_model(cls, model_type, model_path, model_dir, temperature=0.8, context_length=2048, **kwargs) -> RKModel:
        model_classes = {
            RKModelType.LANGUAGE: RKLLM,
            RKModelType.EMBED: RKEMBED,
            RKModelType.RERANKER: RKRERANKER,
        }
        model_cls = model_classes.get(model_type, None)
        if not model_cls:
            return None

        return model_cls(model_path, model_dir, temperature, context_length, **kwargs)
