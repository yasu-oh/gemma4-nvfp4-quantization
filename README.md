# Gemma 4 NVFP4 Quantization

## Overview

Quantizes original Gemma 4 checkpoints and compatible Gemma 4 derived models to NVFP4 with `llmcompressor`.

Calibration uses 1,024 chat-template-normalized samples:

- `CohereLabs/aya_dataset`: 512
- `interstellarninja/hermes_reasoning_tool_use`: 512
  - `single`: 102
  - `multiturn`: 154
  - `multistep`: 205
  - `relevance`: 51

`use_bundled_chat_template` in `config.yaml` selects the bundled, official Gemma template (`true`) or the model default (`false`). Hermes rows with unmatched tool responses are excluded before stratified sampling, then normalized before rendering.

`gemma4-e-nvfp4-quantization.py` is exclusively for Gemma 4 E2B and E4B models.

## Usage
```bash
cd ~
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
cd ~
git clone https://github.com/yasu-oh/gemma4-nvfp4-quantization.git
cd gemma4-nvfp4-quantization
cp sample_config.yaml config.yaml
# Set model_id, device_map, save_dir, and use_bundled_chat_template.
python gemma4-nvfp4-quantization.py
```

Run tests with:

```bash
python -m unittest -v
```
