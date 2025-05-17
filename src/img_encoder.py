import base64
import subprocess
import os
from typing import Optional

import numpy as np


class ImageEncoder:
    def __init__(self, model_path: str, executable_path: str = "./bin/img_encoder"):
        self.model_path = os.path.abspath(model_path)
        self.executable_path = os.path.abspath(executable_path)

        self.process = subprocess.Popen(
            [self.executable_path, self.model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # Unbuffered
            universal_newlines=False
        )

    def encode_image(self, b64_image: str) -> Optional[np.ndarray]:
        image_path = "/tmp/rkllama_image.png"
        image_emb_output = "/tmp/rkllama_image_emb.bin"

        with open(image_path, "wb") as f:
            f.write(base64.b64decode(b64_image))

        try:
            request = f"{image_path}|{image_emb_output}\n".encode('utf-8')
            self.process.stdin.write(request)
            self.process.stdin.flush()

            while True:
                line = self.process.stdout.readline()
                if not line:
                    return None

                if line.startswith(b"Success:"):
                    print(line)
                    with open(image_emb_output, "rb") as f:
                        img_vec = np.fromfile(f, dtype=np.float32)

                    if not img_vec.flags['C_CONTIGUOUS']:
                        img_vec = np.ascontiguousarray(img_vec)

                    return img_vec
                elif line.startswith(b"Error:"):
                    return None
        except Exception as e:
            print(f"Encoding failed: {str(e)}")
            return None

    def __del__(self):
        """Clean up the subprocess."""
        if hasattr(self, 'process'):
            self.process.stdin.close()
            try:
                self.process.wait(timeout=1)
            except Exception:
                self.process.terminate()
            finally:
                print("Image Encoder service stopped")
