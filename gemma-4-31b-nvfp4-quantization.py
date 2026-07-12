#!/usr/bin/env python3

import ast
import json
import re
import warnings
from pathlib import Path
from typing import Any

import yaml

AYA_DATASET = "CohereLabs/aya_dataset"
HERMES_DATASET = "interstellarninja/hermes_reasoning_tool_use"
HERMES_SAMPLES = {
    "single": 102,
    "multiturn": 154,
    "multistep": 205,
    "relevance": 51,
}
SAMPLES_PER_DATASET = 512
SEED = 42

THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
TOOL_CALL = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
TOOL_RESPONSE = re.compile(
    r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL | re.IGNORECASE
)
LEGACY_TOOL_PROMPT = re.compile(
    r"\s*You are a function calling AI model\..*\Z", re.DOTALL | re.IGNORECASE
)
GENERIC_ARRAY = re.compile(r"(?:list|array)\s*\[\s*(.+?)\s*\]", re.IGNORECASE)


class InvalidToolDataError(ValueError):
    pass


class UnmatchedToolResponseError(InvalidToolDataError):
    pass


class InvalidToolCallError(InvalidToolDataError):
    pass


class InvalidToolResponseError(InvalidToolDataError):
    pass


def load_config(work_dir: Path) -> tuple[str, Path, bool]:
    config = yaml.safe_load((work_dir / "config.yaml").read_text(encoding="utf-8"))
    model_id = config["model_id"]
    save_dir = Path(config["save_dir"]).expanduser()
    use_bundled_template = config["use_bundled_chat_template"]
    if not isinstance(use_bundled_template, bool):
        raise TypeError("use_bundled_chat_template must be true or false")
    if not save_dir.is_absolute():
        save_dir = work_dir / save_dir
    return model_id, save_dir, use_bundled_template


def apply_chat_template_file(
    tokenizer: Any, work_dir: Path, use_bundled_template: bool
) -> None:
    if use_bundled_template:
        tokenizer.chat_template = (work_dir / "chat_template.jinja").read_text(
            encoding="utf-8"
        )


def parse(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value.strip())
    except json.JSONDecodeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.literal_eval(value.strip())


def normalize_schema(schema: Any, *, root: bool = False) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "required": []}
    if root and "type" not in schema and "properties" not in schema:
        schema = {"type": "object", "properties": schema}

    result = dict(schema)
    type_name = str(result.get("type", "object" if root else "string")).lower()
    generic = GENERIC_ARRAY.search(type_name)
    if "list" in type_name or "array" in type_name:
        result["type"] = "array"
        items = result.get("items") or ({"type": generic.group(1)} if generic else {})
        result["items"] = normalize_schema(items)
    elif "dict" in type_name or "object" in type_name:
        result["type"] = "object"
        properties = result.get("properties", {})
        result["properties"] = {
            name: normalize_schema(value) for name, value in properties.items()
        }
        result["required"] = result.get("required") or []
    elif "bool" in type_name:
        result["type"] = "boolean"
    elif any(name in type_name for name in ("float", "double", "number")):
        result["type"] = "number"
    elif "int" in type_name:
        result["type"] = "integer"
    else:
        result["type"] = "string"
    return result


def normalize_tools(raw_tools: Any) -> list[dict[str, Any]]:
    normalized = []
    for tool in parse(raw_tools):
        function = tool.get("function", tool)
        definition = {
            "name": function["name"],
            "description": function.get("description", ""),
            "parameters": normalize_schema(function.get("parameters", {}), root=True),
        }
        if "response" in function:
            definition["response"] = normalize_schema(function["response"], root=True)
        normalized.append({"type": "function", "function": definition})
    return normalized


def parse_assistant(
    value: str, turn: int, *, reject_invalid_calls: bool = False
) -> dict[str, Any]:
    reasoning = [text.strip() for text in THINK.findall(value) if text.strip()]
    visible = THINK.sub("", value)
    calls = []
    invalid_calls = []
    for index, raw_call in enumerate(TOOL_CALL.findall(visible)):
        try:
            call = parse(raw_call)
            arguments = parse(call.get("arguments", {}))
            if not isinstance(arguments, dict):
                raise ValueError
        except (AttributeError, KeyError, SyntaxError, TypeError, ValueError):
            if reject_invalid_calls:
                raise InvalidToolCallError(raw_call)
            invalid_calls.append(raw_call.strip())
            continue
        calls.append(
            {
                "id": str(call.get("id") or f"call_{turn}_{index}"),
                "type": "function",
                "function": {"name": call["name"], "arguments": arguments},
            }
        )

    content = TOOL_RESPONSE.sub("", TOOL_CALL.sub("", visible)).strip()
    content = "\n\n".join(part for part in (content, *invalid_calls) if part)
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = "\n\n".join(reasoning)
    if calls:
        message["tool_calls"] = calls
    return message


def extract_string(payload: str, key: str) -> str | None:
    match = re.search(rf'"{key}"\s*:\s*("(?:\\.|[^"\\])*")', payload)
    return json.loads(match.group(1)) if match else None


def parse_response(
    value: str, *, reject_invalid: bool = False
) -> dict[str, Any]:
    payload = value.strip()
    if payload.startswith("**") and payload.endswith("**"):
        payload = payload[2:-2].strip()
    try:
        response = parse(payload)
    except (SyntaxError, ValueError):
        if reject_invalid:
            raise InvalidToolResponseError(value)
        return {
            "content": value,
            "name": extract_string(value, "name"),
            "tool_call_id": extract_string(value, "tool_call_id"),
        }
    if not isinstance(response, dict):
        return {"content": response}
    if "content" not in response and "tool_call_id" not in response:
        return {"content": response}
    return response


def attach_response(messages: list[dict[str, Any]], response: dict[str, Any]) -> bool:
    response_name = response.get("name")
    response_id = response.get("tool_call_id")
    content = response.get("content")

    for message in reversed(messages):
        if any(
            item["name"] == response_name and item["response"] == content
            for item in message.get("tool_responses", [])
        ):
            return True

    for message in reversed(messages):
        calls = message.get("tool_calls", [])
        if not calls:
            continue
        attached = message.setdefault("tool_responses", [])
        used: dict[str, int] = {}
        for item in attached:
            used[item["name"]] = used.get(item["name"], 0) + 1
        seen: dict[str, int] = {}
        pending = []
        for call in calls:
            name = call["function"]["name"]
            seen[name] = seen.get(name, 0) + 1
            if seen[name] > used.get(name, 0):
                pending.append(call)

        if not pending:
            continue

        call = next(
            (
                item
                for item in pending
                if response_id
                and item["id"] == response_id
                and (
                    not response_name
                    or item["function"]["name"] == response_name
                )
            ),
            None,
        )
        if call is None:
            call = next(
                (
                    item
                    for item in pending
                    if response_name
                    and item["function"]["name"] == response_name
                ),
                None,
            )
        if call is None and not response_name and len(pending) == 1:
            call = pending[0]
        if call is None:
            return False

        if response_id:
            call["id"] = response_id
        attached.append(
            {"name": call["function"]["name"], "response": content}
        )
        return True
    return False


def normalize_conversation(
    conversation: Any, *, strict_tool_data: bool = False
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(conversation):
        role, value = turn["from"], turn.get("value", "")
        if role == "system":
            content = LEGACY_TOOL_PROMPT.sub("", value).strip()
            if content:
                messages.append({"role": "system", "content": content})
        elif role == "human":
            messages.append({"role": "user", "content": value.strip()})
        elif role == "gpt":
            assistant = parse_assistant(
                value,
                turn_index,
                reject_invalid_calls=strict_tool_data,
            )
            messages.append(assistant)
            for raw_response in TOOL_RESPONSE.findall(THINK.sub("", value)):
                response = parse_response(
                    raw_response, reject_invalid=strict_tool_data
                )
                if not attach_response(messages, response):
                    if strict_tool_data and (
                        response.get("name") or response.get("tool_call_id")
                    ):
                        raise UnmatchedToolResponseError(raw_response)
                    fallback = str(response.get("content") or "").strip("* \n")
                    assistant["content"] = "\n\n".join(
                        part
                        for part in (assistant["content"].strip("* \n"), fallback)
                        if part
                    )
        elif role == "tool":
            for raw_response in TOOL_RESPONSE.findall(value) or [value]:
                response = parse_response(
                    raw_response, reject_invalid=strict_tool_data
                )
                if not attach_response(messages, response):
                    if strict_tool_data:
                        raise UnmatchedToolResponseError(raw_response)
        else:
            raise ValueError(f"Unsupported role: {role}")
    return messages


def has_consistent_tool_data(conversation: Any) -> bool:
    try:
        normalize_conversation(conversation, strict_tool_data=True)
    except InvalidToolDataError:
        return False
    return True


def select_hermes_samples(dataset: Any) -> Any:
    from datasets import concatenate_datasets

    dataset = dataset.filter(
        lambda row: has_consistent_tool_data(row["conversations"])
    )
    subsets = []
    for category, count in HERMES_SAMPLES.items():
        subset = dataset.filter(
            lambda row, expected=category: row["scenario_category"] == expected
        )
        if len(subset) < count:
            raise ValueError(f"Not enough Hermes rows for {category}")
        subsets.append(subset.shuffle(seed=SEED).select(range(count)))
    return concatenate_datasets(subsets).shuffle(seed=SEED)


def main() -> None:
    from datasets import concatenate_datasets, load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    work_dir = Path.cwd()
    model_id, save_dir, use_bundled_template = load_config(work_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    apply_chat_template_file(tokenizer, work_dir, use_bundled_template)

    aya = load_dataset(AYA_DATASET, split="train").shuffle(seed=SEED).select(
        range(SAMPLES_PER_DATASET)
    )

    def format_aya(row: dict[str, Any]) -> dict[str, str]:
        messages = [
            {"role": "user", "content": row["inputs"].strip()},
            {"role": "assistant", "content": row["targets"].strip()},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }

    hermes = select_hermes_samples(load_dataset(HERMES_DATASET, split="train"))

    def format_hermes(row: dict[str, Any]) -> dict[str, str]:
        return {
            "text": tokenizer.apply_chat_template(
                normalize_conversation(
                    row["conversations"], strict_tool_data=True
                ),
                tools=normalize_tools(row["tools"]),
                tokenize=False,
                add_generation_prompt=False,
                preserve_thinking=True,
            )
        }

    calibration_dataset = concatenate_datasets(
        [
            aya.map(format_aya, remove_columns=aya.column_names),
            hermes.map(format_hermes, remove_columns=hermes.column_names),
        ]
    ).shuffle(seed=SEED)

    recipe = QuantizationModifier(
        targets=["Linear"],
        scheme="NVFP4",
        ignore=["re:.*vision.*", "re:.*audio.*", "lm_head", "re:.*embed.*"],
    )
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset=calibration_dataset,
        num_calibration_samples=SAMPLES_PER_DATASET * 2,
        shuffle_calibration_samples=False,
        text_column="text",
        max_seq_length=8192,
        pad_to_max_length=False,
    )
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
