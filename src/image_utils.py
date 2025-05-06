import base64
import subprocess

import numpy as np

def read_img_emb(b64_image, image_emb_model_path = None, image_encoder_bin = None):
    if image_emb_model_path is None:
        # Not a VL model
        return None

    image_path = "/tmp/rkllama_image.png"
    image_emb_output = "/tmp/rkllama_image_emb.bin"

    with open(image_path, "wb") as f:
        f.write(base64.b64decode(b64_image))

    args = [
        image_emb_model_path, image_path, image_emb_output
    ]

    result = subprocess.run(
        [image_encoder_bin] + args,
        capture_output=True,
        text=True
    )
    print("imgenc output: ", result.stdout)
    if result.returncode != 0:
        # Failed to embedding
        return None
    
    with open(image_emb_output, "rb") as f:
        img_vec = np.fromfile(f, dtype=np.float32)
    
    if not img_vec.flags['C_CONTIGUOUS']:
        img_vec = np.ascontiguousarray(img_vec)

    return img_vec

