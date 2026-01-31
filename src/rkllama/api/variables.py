import threading
from dataclasses import dataclass
import numpy as np
from rkllama.config import is_debug_mode
from rkllama.api.worker import WorkerManager


@dataclass
class LogitsResult:
    logits: np.ndarray
    vocab_size: int
    num_tokens: int


isLocked = False
global_status = -1
global_text = []
global_gui_actor_result = None
global_rerank_logits: LogitsResult = None
split_byte_data = bytes(b"")

# Worker variables
worker_manager_rkllm = WorkerManager()


verrou = threading.Lock()

model_id = ""
system = "Tu es un assistant artificiel."
model_config = {}  # For storing model-specific configuration
generation_complete = False  # Flag to track completion status
debug_mode = is_debug_mode()
stream_stats = {
    "total_requests": 0,
    "successful_responses": 0,
    "failed_responses": 0,
    "incomplete_streams": 0  # Streams that didn't receive done=true
}