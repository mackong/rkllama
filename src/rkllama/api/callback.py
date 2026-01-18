import ctypes, sys
import numpy as np
from .classes import *
import rkllama.api.variables as variables

last_embeddings = []

# Definir la fonction de rappel
def callback_impl(result, userdata, status):
    global last_embeddings
    if status == LLMCallState.RKLLM_RUN_FINISH:
        variables.global_status = status
        print("\n")
        sys.stdout.flush()
    elif status == LLMCallState.RKLLM_RUN_ERROR:
        variables.global_status = status
        print("Execution Error")
        sys.stdout.flush()
    elif status == LLMCallState.RKLLM_RUN_NORMAL:
        # Sauvegarder le texte du token de sortie et l'status d'execution de RKLLM
        variables.global_status = status
        # Check if result or result.contents or result.contents.text is None
        try:
            # Add defensive checks to prevent None concatenation
            if result and result.contents and result.contents.text:
                text_bytes = result.contents.text
                if not isinstance(text_bytes, bytes):
                    # If not bytes, try to convert or use empty bytes
                    try:
                        text_bytes = bytes(text_bytes)
                    except:
                        text_bytes = b""
                        
                # Now safely concatenate
                try:
                    decoded_text = (variables.split_byte_data + text_bytes).decode('utf-8')
                    variables.global_text.append(decoded_text)
                    print(decoded_text, end='')
                    variables.split_byte_data = bytes(b"")
                except UnicodeDecodeError:
                    # Handle incomplete UTF-8 sequences
                    variables.split_byte_data += text_bytes
            else:
                # Handle case where text is None
                if variables.split_byte_data:
                    try:
                        # Try to decode any accumulated bytes
                        decoded_text = variables.split_byte_data.decode('utf-8')
                        variables.global_text.append(decoded_text)
                        print(decoded_text, end='')
                        variables.split_byte_data = bytes(b"")
                    except UnicodeDecodeError:
                        # Still incomplete, keep for next time
                        pass
            
            # --- EMBEDDINGS Part---
            if result and result.contents and result.contents.last_hidden_layer.hidden_states:
                num_tokens = result.contents.last_hidden_layer.num_tokens
                embd_size = result.contents.last_hidden_layer.embd_size
                total_size = num_tokens * embd_size

                # Convert pointer to numpy
                array_type = ctypes.c_float * total_size
                raw = array_type.from_address(
                    ctypes.addressof(result.contents.last_hidden_layer.hidden_states.contents)
                )
                embeddings = np.ctypeslib.as_array(raw)
                embeddings = embeddings.reshape(num_tokens, embd_size)

                # Save global
                #last_embeddings = embeddings.copy()
                last_embeddings.append(embeddings)
                print(f"\n✅ Embeddings Shape: {embeddings.shape}")

        except Exception as e:
            print(f"\nError processing callback: {str(e)}", end='')

        sys.stdout.flush()


def gui_actor_callback_impl(result_ptr, userdata_ptr, state):
    """Callback for GUI Actor inference"""
    if state == LLMCallState.RKLLM_RUN_NORMAL:
        variables.global_status = state
        result = result_ptr.contents
        last_hidden_layer = result.last_hidden_layer
        if last_hidden_layer.hidden_states and last_hidden_layer.embd_size > 0:
            hidden_size = last_hidden_layer.embd_size
            num_tokens = last_hidden_layer.num_tokens
            if num_tokens > 0:
                hidden_array = np.ctypeslib.as_array(
                    last_hidden_layer.hidden_states,
                    shape=(num_tokens, hidden_size)
                ).copy()
                variables.global_gui_actor_result = hidden_array
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        variables.global_status = state
        print("erreur d'execution")
        sys.stdout.flush()


def embed_callback_impl(result_ptr, userdata_ptr, state):
    """Callback for embedding inference"""
    if state == LLMCallState.RKLLM_RUN_NORMAL:
        variables.global_status = state
        result = result_ptr.contents
        last_hidden_layer = result.last_hidden_layer
        if last_hidden_layer.hidden_states and last_hidden_layer.embd_size > 0:
            embd_size = last_hidden_layer.embd_size
            num_tokens = last_hidden_layer.num_tokens
            if num_tokens > 0:
                last_token_embedding = np.array([
                    last_hidden_layer.hidden_states[(num_tokens - 1) * embd_size + i]
                    for i in range(embd_size)
                ])
                variables.global_embed = variables.EmbedResult(
                    embedding=last_token_embedding,
                    embd_size=embd_size,
                    num_tokens=num_tokens
                )
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        variables.global_status = state
        print("erreur d'execution")
        sys.stdout.flush()


def rerank_callback_impl(result_ptr, userdata_ptr, state):
    """Callback for reranking inference"""
    if state == LLMCallState.RKLLM_RUN_NORMAL:
        variables.global_status = state
        result = result_ptr.contents
        logits = result.logits
        if logits.logits and logits.vocab_size > 0:
            vocab_size = logits.vocab_size
            num_tokens = logits.num_tokens
            if num_tokens > 0:
                last_logits = np.array([
                    logits.logits[(num_tokens - 1) * vocab_size + i]
                    for i in range(vocab_size)
                ])
                variables.global_rerank_logits = variables.LogitsResult(
                    logits=last_logits,
                    vocab_size=vocab_size,
                    num_tokens=num_tokens
                )
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        variables.global_status = state
        print("erreur d'execution")
        sys.stdout.flush()    