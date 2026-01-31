import ctypes, sys
import numpy as np
from .classes import *
from .variables import *
import rkllama.api.variables as variables

global_status = -1
global_text = []
split_byte_data = bytes(b"")
last_embeddings = []
last_rerank = []
last_gui_actor = []

# Definir la fonction de rappel
def callback_impl(result, userdata, status):
    global split_byte_data, global_status, global_text, last_embeddings, last_rerank, last_gui_actor

    if status == LLMCallState.RKLLM_RUN_FINISH:
        global_status = status
        print("\n")
        sys.stdout.flush()
    elif status == LLMCallState.RKLLM_RUN_ERROR:
        global_status = status
        print("Execution Error")
        sys.stdout.flush()
    elif status == LLMCallState.RKLLM_RUN_NORMAL:
        # Sauvegarder le texte du token de sortie et l'status d'execution de RKLLM
        global_status = status
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
                    decoded_text = (split_byte_data + text_bytes).decode('utf-8')
                    global_text.append(decoded_text)
                    print(decoded_text, end='')
                    split_byte_data = bytes(b"")
                except UnicodeDecodeError:
                    # Handle incomplete UTF-8 sequences
                    split_byte_data += text_bytes
            else:
                # Handle case where text is None
                if split_byte_data:
                    try:
                        # Try to decode any accumulated bytes
                        decoded_text = split_byte_data.decode('utf-8')
                        global_text.append(decoded_text)
                        print(decoded_text, end='')
                        split_byte_data = bytes(b"")
                    except UnicodeDecodeError:
                        # Still incomplete, keep for next time
                        pass

            # --- EMBEDDINGS Part (also used by GUI Actor) ---
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

                # Save global for both embeddings and GUI Actor (they use same hidden states)
                #last_embeddings = embeddings.copy()
                last_embeddings.append(embeddings)
                last_gui_actor.append(embeddings)
                print(f"\n✅ Hidden Layer Shape: {embeddings.shape}")

            # --- LOGITS Part (for Rerank) ---
            if result and result.contents:
                try:
                    logits_info = result.contents.logits
                    if logits_info and logits_info.logits:
                        vocab_size = logits_info.vocab_size
                        num_tokens = logits_info.num_tokens
                        if num_tokens > 0 and vocab_size > 0:
                            # Extract last token logits
                            logits_array = np.array([
                                logits_info.logits[(num_tokens - 1) * vocab_size + i]
                                for i in range(vocab_size)
                            ])
                            # Store in module-level list for worker to access (like last_embeddings)
                            last_rerank.append(logits_array)
                            print(f"\n✅ Logits captured: vocab_size={vocab_size}, num_tokens={num_tokens}")
                except (AttributeError, TypeError) as e:
                    # Logits might not be available in this mode
                    pass

        except Exception as e:
            print(f"\nError processing callback: {str(e)}", end='')

        sys.stdout.flush()