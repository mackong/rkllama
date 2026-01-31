# img_encoder.py
import numpy as np
import logging
import cv2
import base64
import requests
from rknnlite.api.rknn_lite import RKNNLite
import os

logger = logging.getLogger("rkllama.rknnlite")


# Run the visionencder to get the image embedding
def run_vision_encoder(model_encoder_path, images_source, image_width, image_height):
    """
    Run the vision encoder to get the image embedding
    Args:
        model_encoder_path (str): Path of the encoder
        image_source (str): Image source (base64, PATH or URL)
        image_width (int): Width of the image
        image_height (int): Height of the image
    Returns:
        np.ndarray: Image embedding
    """

    # Prepare the image
    prepared_images = [prepare_image(image_source, image_width, image_height) for image_source in images_source]

    # Init encoder
    vision_encoder = RKNNLite(verbose=False)
    vision_encoder.load_rknn(model_encoder_path)
    vision_encoder.init_runtime()

    # Inference
    image_embeddings = [vision_encoder.inference(inputs=[img.astype(np.float32)], data_type="float32", data_format="nhwc")[0] for img in prepared_images]
    logger.debug(f"Image embeddings shapes: {[emb.shape for emb in image_embeddings]}")

    # Concatenate along the first axis (rows)
    np_float32_image_embeddings = [emb.astype(np.float32) for emb in image_embeddings]
    concatenated_image_embeddings = np.concatenate(np_float32_image_embeddings, axis=0)
    logger.debug(f"Concatenated image embeddings shape: {concatenated_image_embeddings.shape}")

    # Release RKNNLite resources
    vision_encoder.release()

    # Return the encoded image to the main process
    return concatenated_image_embeddings


def load_image(source: str):
    """
    Load an image from:
      - a local path
      - a URL
      - a Base64 string
    Returns:
      - image as numpy array (BGR) or None if fails
    """
    img = None

    # Case 1: local file
    if os.path.exists(source):
        img = cv2.imread(source)

    # Case 2: URL
    elif source.startswith("http://") or source.startswith("https://"):
        try:
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error("Error loading from URL:", e)

    # Case 3: Base64
    else:
        try:
            # Remove "data:image/..;base64," if present
            if "," in source:
                source = source.split(",")[1]
            img_data = base64.b64decode(source)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error("Error loading from Base64:", e)

    # Convert BGR → RGB (Color fix)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def prepare_image(image_source, image_width, image_height) -> np.ndarray:
    """ Load and preprocess an image for model input.
        Args:
            image_source: Path, URL, or Base64 string of the image.
            image_width: Target width for resizing.
            image_height: Target height for resizing.
        Returns:
            Preprocessed image as a numpy array (HWC, uint8).
    """
    # Read image
    img = load_image(image_source)  # BGR
    if img is None:
        raise FileNotFoundError(image_source)

    # Preprocess Image
    resized = cv2.resize(img, (image_width, image_height))
    resized = resized.astype(np.float32)
    resized = resized[np.newaxis, :, :, :]

    # Return the image
    return resized


def letterbox(img, new_shape, color=(127.5, 127.5, 127.5)):
    """
    Resize and pad image while preserving aspect ratio (used for GUI Actor)

    Args:
        img: Input image
        new_shape: Target shape (height, width)
        color: Padding color

    Returns:
        tuple: (padded_image, ratio, (dw, dh))
    """
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
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def prepare_image_for_gui_actor(image_source, image_width, image_height):
    """
    Load and preprocess an image for GUI Actor with letterbox processing

    Args:
        image_source: Path, URL, or Base64 string of the image.
        image_width: Target width for resizing.
        image_height: Target height for resizing.

    Returns:
        tuple: (preprocessed_image, ratio, dw, dh)
    """
    # Read image
    img = load_image(image_source)  # RGB
    if img is None:
        raise FileNotFoundError(image_source)

    # Apply letterbox to preserve aspect ratio
    img_letterboxed, ratio, (dw, dh) = letterbox(img, (image_height, image_width))

    input_tensor = img_letterboxed.astype(np.float32)
    input_tensor = (input_tensor / 255.0 - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

    return input_tensor, ratio, dw, dh


def run_gui_actor_vision_encoder(model_encoder_path, images_source, image_width, image_height):
    """
    Run the GUI Actor vision encoder to get the image embedding with letterbox processing

    Args:
        model_encoder_path (str): Path of the encoder
        images_source (list): List of image sources (base64, PATH or URL)
        image_width (int): Width of the image
        image_height (int): Height of the image

    Returns:
        dict: Dictionary containing image embeddings and letterbox parameters
            {
                'embeddings': np.ndarray,
                'ratio': float,
                'dw': float,
                'dh': float
            }
    """
    # Prepare images with letterbox processing (GUI Actor only supports single image)
    if len(images_source) > 1:
        logger.warning(f"GUI Actor vision encoder received {len(images_source)} images, using only the first one")

    image_source = images_source[0]
    img_prepared, ratio, dw, dh = prepare_image_for_gui_actor(image_source, image_width, image_height)

    # Init encoder
    vision_encoder = RKNNLite(verbose=False)
    vision_encoder.load_rknn(model_encoder_path)
    vision_encoder.init_runtime()

    # Inference
    image_embedding = vision_encoder.inference(inputs=[img_prepared.astype(np.float32)], data_type="float32", data_format="nhwc")[0]
    logger.debug(f"GUI Actor image embedding shape: {image_embedding.shape}")

    # Convert to float32
    image_embedding = image_embedding.astype(np.float32)

    # Release RKNNLite resources
    vision_encoder.release()

    # Return the encoded image with letterbox parameters
    return {
        'embeddings': image_embedding,
        'ratio': ratio,
        'dw': dw,
        'dh': dh
    }


def base64_to_cv2(image_data):
    """Convert base64 string to cv2 image"""
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def cv2_to_base64(image):
    """Convert cv2 image to base64 string"""
    _, im_arr = cv2.imencode(".jpg", image)
    im_bytes = im_arr.tobytes()
    return base64.b64encode(im_bytes).decode("utf-8")


def get_prediction_point(attn_scores, n_width, n_height, activation_threshold=0.3):
    """
    Calculate best prediction point from attention scores

    Args:
        attn_scores: Attention scores from pointer head
        n_width: Number of width patches
        n_height: Number of height patches
        activation_threshold: Threshold for activation

    Returns:
        tuple: (center_x, center_y) normalized coordinates
    """
    # Get first element if batch dimension exists
    if len(attn_scores.shape) > 1:
        attn_scores = attn_scores[0]

    max_score = np.max(attn_scores)
    threshold = max_score * activation_threshold
    mask = attn_scores > threshold
    valid_indices = np.where(mask)[0]

    if valid_indices.size == 0:
        # No activation point found, return center
        return (0.5, 0.5)

    # Calculate weighted average
    scores = attn_scores[valid_indices]
    total_score = np.sum(scores)

    y_coords = valid_indices.astype(np.float32) // n_width
    x_coords = valid_indices.astype(np.float32) % n_width

    center_x = np.sum(((x_coords + 0.5) / n_width) * scores) / total_score
    center_y = np.sum(((y_coords + 0.5) / n_height) * scores) / total_score

    return (float(center_x), float(center_y))


def run_gui_actor_pointer_head(
    pointer_head_path,
    hidden_states,
    image_embeddings,
    ratio,
    dw,
    dh,
    input_image_data=None,
    image_width=1148,
    image_height=896,
    label=False
):
    """
    Run the GUI Actor pointer head to predict click coordinates

    Args:
        pointer_head_path (str): Path to the pointer head RKNN model
        hidden_states: Hidden states from RKLLM model
        image_embeddings: Image embeddings from vision encoder
        ratio (float): Letterbox ratio from vision encoder
        dw (float): Letterbox width padding
        dh (float): Letterbox height padding
        input_image_data (str, optional): Base64 encoded input image for labeling
        image_width (int): Width used for vision encoding (default: 1148)
        image_height (int): Height used for vision encoding (default: 896)
        label (bool): Whether to draw marker on output image

    Returns:
        dict: Dictionary containing prediction results
            {
                'px': int,  # Predicted x coordinate
                'py': int,  # Predicted y coordinate
                'labeled_image': str or None  # Base64 encoded image with marker if label=True
            }
    """
    # Prepare inputs
    decoder_hidden_states = hidden_states[-2].reshape(1, -1).astype(np.float32)
    image_embeds = image_embeddings.astype(np.float32)
    logger.debug(f"Pointer head inputs shapes: enc={image_embeds.shape}, dec={decoder_hidden_states.shape}")

    # Init pointer head
    pointer_head = RKNNLite(verbose=False)
    ret = pointer_head.load_rknn(pointer_head_path)
    if ret != 0:
        logger.error(f"Failed to load pointer head from {pointer_head_path}")
        return {'px': 0, 'py': 0, 'labeled_image': None}

    ret = pointer_head.init_runtime()
    if ret != 0:
        logger.error("Failed to init pointer head runtime")
        pointer_head.release()
        return {'px': 0, 'py': 0, 'labeled_image': None}

    # Run pointer head inference
    outputs = pointer_head.inference(inputs=[image_embeds, decoder_hidden_states])
    attn_scores = outputs[0]
    logger.debug(f"Got attention scores, shape: {attn_scores.shape}")

    # Release RKNNLite resources
    pointer_head.release()

    # Calculate prediction point
    n_height = image_height / 28
    n_width = image_width / 28
    best_point = get_prediction_point(attn_scores, n_width, n_height)
    logger.debug(f"Predicted Best Point (normalized): x={best_point[0]:.4f}, y={best_point[1]:.4f}")

    # Convert normalized coordinates to image coordinates
    pred_x_padded = best_point[0] * image_width
    pred_y_padded = best_point[1] * image_height

    top = int(round(dh - 0.1))
    left = int(round(dw - 0.1))

    # Default coordinates if no image data provided
    px, py = 0, 0
    labeled_image = None

    if input_image_data:
        img = base64_to_cv2(input_image_data)
        h, w, _ = img.shape

        px = int((pred_x_padded - left) / ratio)
        py = int((pred_y_padded - top) / ratio)

        px = int(np.clip(px, 0, w - 1))
        py = int(np.clip(py, 0, h - 1))

        if label:
            cv2.drawMarker(img, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 25, 1)
            labeled_image = cv2_to_base64(img)

    return {
        'px': px,
        'py': py,
        'labeled_image': labeled_image
    }


