#!/usr/bin/env python3

from pathlib import Path
from typing import Any

import yaml

from calibration_dataset import create_calibration_dataset


def load_config(work_dir: Path) -> tuple[str, Any, Path, bool]:
    config = yaml.safe_load((work_dir / "config.yaml").read_text(encoding="utf-8"))
    model_id = config["model_id"]
    device_map = config["device_map"]
    save_dir = Path(config["save_dir"]).expanduser()
    use_bundled_template = config["use_bundled_chat_template"]
    if not isinstance(use_bundled_template, bool):
        raise TypeError("use_bundled_chat_template must be true or false")
    if not save_dir.is_absolute():
        save_dir = work_dir / save_dir
    return model_id, device_map, save_dir, use_bundled_template


def apply_chat_template_file(
    tokenizer: Any, work_dir: Path, use_bundled_template: bool
) -> None:
    if use_bundled_template:
        tokenizer.chat_template = (work_dir / "chat_template.jinja").read_text(
            encoding="utf-8"
        )


def main() -> None:
    from compressed_tensors.quantization import QuantizationArgs, preset_name_to_scheme
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    work_dir = Path.cwd()
    model_id, device_map, save_dir, use_bundled_template = load_config(work_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    apply_chat_template_file(tokenizer, work_dir, use_bundled_template)

    calibration_dataset = create_calibration_dataset(tokenizer)

    ignore = [
        "lm_head",
        "re:.*embed.*",
        "re:.*vision.*",
        "re:.*audio.*",
        "re:.*router.*",
    ]

    nvfp4_scheme = preset_name_to_scheme("NVFP4", ["Linear"])

    if nvfp4_scheme.weights is None:
        raise RuntimeError("NVFP4 preset does not define weight quantization")

    nvfp4_scheme.weights.observer = "imatrix_mse"
    nvfp4_scheme.weights.observer_kwargs = {
        "strict": True,
        "expand": 1.8,
        "maxshrink": 1 - 0.8 / 1.8,
        "grid": 200.0,
        "norm": 2.4,
        "patience": 1000,
    }

    recipe = [
        GPTQModifier(
            config_groups={
                "group_0": nvfp4_scheme,
            },
            ignore=ignore,
            kv_cache_scheme=QuantizationArgs(
                num_bits=8,
                type="float",
                symmetric=True,
                strategy="tensor",
                dynamic=False,
                observer="static_minmax",
            ),
        ),
    ]

    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset=calibration_dataset,
        num_calibration_samples=len(calibration_dataset),
        shuffle_calibration_samples=False,
        text_column="text",
        max_seq_length=8192,
        pad_to_max_length=False,
    )
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
