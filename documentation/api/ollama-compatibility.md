# RKLLAMA - Ollama API Compatibility

## Overview

RKLLAMA now implements an Ollama-compatible API, providing an interface that matches key Ollama endpoints. This enables integration with many tools and scripts designed for Ollama's API, particularly those using the chat and generate functionality.

## Supported Endpoints

| Endpoint | Method | Description | Status |
|----------|--------|-------------|--------|
| `/api/tags` | GET | List available models | ✅ |
| `/api/version` | GET | Get API version (Dummy version to fix some apps) | ✅ |
| `/api/show` | POST | Show model information | ✅ |
| `/api/create` | POST | Create model from Modelfile | ⚠️ Basic implementation |
| `/api/pull` | POST | Pull a model | ⚠️ Basic implementation |
| `/api/delete` | DELETE | Delete a model | ✅ |
| `/api/generate` | POST | Generate a completion | ✅ |
| `/api/chat` | POST | Generate a chat completion | ✅ |
| `/api/embeddings` | POST | Generate embeddings | ✅ |
| `/api/embed` | POST | Generate embeddings | ❌ Not implemented |
| `/api/rerank` | POST | Rerank document scores | ✅ |
| `/api/gui_actor` | POST | Gui Actor | ✅ |

## Usage Examples

### Chat Completion (`/api/chat`)

This endpoint uses a chat-style format with message roles and is best for multi-turn conversations:

```bash
curl -X POST http://localhost:8080/api/chat -d '{
  "model": "qwen2.5:3b",
  "messages": [
    {"role": "user", "content": "Hello, how are you today?"}
  ]
}'
```

### Generate Completion (`/api/generate`)

This endpoint is used for single-turn completions based on a prompt:

```bash
curl -X POST http://localhost:8080/api/generate -d '{
  "model": "qwen2.5:3b",
  "prompt": "Write a poem about AI"
}'
```

### Generate Embeddings (`/api/embeddings`)

This endpoint is used for generate embeddings for a prompt:

```bash
curl -X POST http://localhost:8080/api/embeddings -d '{
  "model": "Qwen3-Embedding-0.6B_W8A8_RK3588",
  "prompt": "What is the capital of China?"
}'
```

Response like:
```json
{
  "embedding": [
    -0.07492150782553514,
    -0.0005481026958273948,
    -0.005530112881487222,
    -0.004405116877649514,
    0.0013207140529182097,
    -0.0246007981018471,
    0.0388100613696313,
    -0.10218408203542528,
    -0.07558312074362279,
    -0.011990026536975796
  ]
}
```

*NOTE*: In Modelfile of embedding models, must setting `MODEL_TYPE="embed"`.

### Rerank Document Scores (`/api/rerank`)

This endpoint is used for rerank document scores:

```bash
curl -X POST http://localhost:8080/api/rerank -d '{
  "model": "Qwen3-Reranker-0.6B_W8A8_RK3588",
  "prompt": "What is the capital of China?",
  "documents": [
      "The capital of China is Beijing.",
      "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
  ]
}'
```

Response like:
```json
{
  "scores": [
    {
      "document": "The capital of China is Beijing.",
      "score": 0.9950627462305758
    },
    {
      "document": "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
      "score": 0.011146341665756624
    }
  ]
}
```

*NOTE*: In Modelfile of reranker models, must setting `MODEL_TYPE="reranker"`.

### Gui Actor (`/api/gui_actor`)

This endpoint is used for gui actor:

```bash
curl -X POST http://localhost:8080/api/gui_actor -d "{
  \"model\": \"Qwen2.5-VL-3B_W8A8_RK3588\",
  \"prompt\": \"refresh dir\",
  \"image\": \"$(base64 -w 0 image.jpg)\",
  \"label\": true
}"
```

the label is False by default.

Response like this if label is True:
```json
{
  "image": "base64 image data",
  "px": 100,
  "py": 100
}
```

Response like this if label is False:
```json
{
  "image": None,
  "px": 100,
  "py": 100
}
```

*NOTE*: In Modelfile of gui actor models, must setting `VISION_ENCODER_PATH="vision_encoder.rknn"` and `POINTER_HEAD_PATH="pointer_head.rknn"`.

### List Models

```bash
curl http://localhost:8080/api/tags
```

## Platform Auto-detection

RKLLAMA automatically detects whether you're using an RK3588 or RK3576 platform. If detection fails, you'll be prompted to select your CPU model:

```
CPU model not detected automatically.
Please select your CPU model:
1) rk3588
2) rk3576
Enter selection (1-2):
```

## Important Differences from Ollama

1. **Model Format**: RKLLAMA uses `.rkllm` files optimized for Rockchip NPUs, not Ollama's format
2. **Modelfile Requirements**: RKLLAMA Modelfiles require a `HUGGINGFACE_PATH` parameter
3. **NPU Acceleration**: RKLLAMA runs on NPU hardware rather than CPU/GPU
4. **External Tokenizers**: RKLLAMA uses HuggingFace tokenizers fetched at runtime

## Optional Debugging Mode

For troubleshooting purposes, RKLLAMA includes an optional debug mode that can be enabled when starting the server:

```bash
# Using the rkllama command
rkllama serve --debug

# Or directly with the server script
bash ~/RKLLAMA/server.sh --debug
```

When debug mode is enabled:
- Detailed logs are written to `~/RKLLAMA/rkllama_debug.log`
- Additional diagnostic information is displayed in the console
- A special `/api/debug` endpoint becomes available (advanced users only)

Debug mode is entirely optional and not needed for normal operation.

## Stream Reliability Improvements

Recent updates have significantly improved streaming reliability:
- Enhanced "done" signaling for proper stream completion
- Fixed token tracking across streaming sessions

## Limitations

- Only core endpoints (`/api/chat` and `/api/generate`) are fully implemented
- Pull and Create endpoints have basic implementations
- Embeddings API have basic implementations
- Some advanced Ollama formatting features are not yet supported
- Not all Ollama clients have been tested for compatibility
- OpenAI API compatibility is still in development
- Additional Rerank API have basic implementations

## Troubleshooting Tips

- Check if server is running and accessible
- Verify models are properly loaded before making requests
- Try non-streaming requests (`"stream": false`) if streaming has issues
- Ensure your Modelfile includes required `HUGGINGFACE_PATH` parameter
- For advanced troubleshooting, enable debug mode with `--debug` flag
