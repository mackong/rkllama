import ctypes
import sys
import time
import numpy as np
import src.variables as variables
from .classes import *


def callback_impl(resultat, donnees_utilisateur, etat):
    """Definir la fonction de rappel."""
    if etat == LLMCallState.RKLLM_RUN_FINISH:
        variables.global_status = etat
        print("\n")
        sys.stdout.flush()
    elif etat == LLMCallState.RKLLM_RUN_ERROR:
        variables.global_status = etat
        print("erreur d'execution")
        sys.stdout.flush()
    elif etat == LLMCallState.RKLLM_RUN_GET_LAST_HIDDEN_LAYER:
        '''
        Si vous utilisez la fonction GET_LAST_HIDDEN_LAYER, l'interface de rappel renverra le pointeur de memoire : last_hidden_layer,
        le nombre de tokens : num_tokens, et la taille de la couche cachee : embd_size.
        Avec ces trois parametres, vous pouvez recuperer les donnees de last_hidden_layer.
        Remarque : Les donnees doivent etre recuperees pendant le rappel actuel ; si elles ne sont pas obtenues a temps, le pointeur sera libere lors du prochain rappel.
        '''
        if resultat.last_hidden_layer.embd_size != 0 and resultat.last_hidden_layer.num_tokens != 0:
            taille_donnees = resultat.last_hidden_layer.embd_size * resultat.last_hidden_layer.num_tokens * ctypes.sizeof(ctypes.c_float)
            print(f"taille_donnees : {taille_donnees}")
            
            variables.global_text.append(f"taille_donnees : {taille_donnees}\n")
            chemin_sortie = os.getcwd() + "/last_hidden_layer.bin"

            with open(chemin_sortie, "wb") as fichier_sortie:
                donnees = ctypes.cast(resultat.last_hidden_layer.hidden_states, ctypes.POINTER(ctypes.c_float))
                type_tableau_float = ctypes.c_float * (taille_donnees // ctypes.sizeof(ctypes.c_float))
                tableau_float = type_tableau_float.from_address(ctypes.addressof(donnees.contents))
                fichier_sortie.write(bytearray(tableau_float))
                print(f"Donnees sauvegardees dans {chemin_sortie} avec succes !")
                variables.global_text.append(f"Donnees sauvegardees dans {chemin_sortie} avec succes !")
        else:
            print("Donnees de la couche cachee invalides.")
            variables.global_text.append("Donnees de la couche cachee invalides.")
        
        variables.global_status = etat
        time.sleep(0.05)  # Attente de 0,05 seconde pour attendre le resultat de la sortie
        sys.stdout.flush()
    else:
        # Sauvegarder le texte du token de sortie et l'etat d'execution de RKLLM
        variables.global_status = etat
        # Check if resultat or resultat.contents or resultat.contents.text is None
        try:
            # Add defensive checks to prevent None concatenation
            if resultat and resultat.contents and resultat.contents.text:
                text_bytes = resultat.contents.text
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
        except Exception as e:
            print(f"\nError processing callback: {str(e)}", end='')
            
        sys.stdout.flush()


def gui_actor_callback_impl(result_ptr, userdata_ptr, state):
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
