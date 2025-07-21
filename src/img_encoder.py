import base64
import subprocess
import os
from typing import Optional

import numpy as np

import config


class ImageEncoder:
    def __init__(self, model_path: str):
        self.model_path = os.path.abspath(model_path)

        exe_path = os.path.join(config.get_path("bin"), "img_encoder")

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = config.get_path("lib")

        self.process = subprocess.Popen(
            [exe_path, self.model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # Unbuffered
            universal_newlines=False,
            env=env,
        )

    def encode_image(self, b64_image: str) -> Optional[np.ndarray]:
        image_path = "/tmp/rkllama_image.png"
        image_emb_output = "/tmp/rkllama_image_emb.bin"

        with open(image_path, "wb") as f:
            f.write(base64.b64decode(b64_image))

        try:
            request = f"{image_path}|{image_emb_output}\n".encode("utf-8")
            self.process.stdin.write(request)
            self.process.stdin.flush()

            while True:
                line = self.process.stdout.readline()
                if not line:
                    return None

                line = line.decode("utf-8")
                if line.startswith("Success:"):
                    print(line)
                    with open(image_emb_output, "rb") as f:
                        img_vec = np.fromfile(f, dtype=np.float32)

                    if not img_vec.flags["C_CONTIGUOUS"]:
                        img_vec = np.ascontiguousarray(img_vec)

                    return img_vec
                elif line.startswith("Error:"):
                    return None
        except Exception as e:
            print(f"Encoding failed: {str(e)}")
            return None

    def release(self):
        """Clean up the subprocess."""
        if hasattr(self, 'process'):
            self.process.stdin.close()
            try:
                self.process.kill()
                self.process.wait(1)
            finally:
                print("Image Encoder service stopped")
