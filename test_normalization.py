import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).parent
GENERAL_SCRIPT = ROOT / "gemma4-nvfp4-quantization.py"
E_SCRIPT = ROOT / "gemma4-e-nvfp4-quantization.py"
DATASET_SCRIPT = ROOT / "calibration_dataset.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERAL = load_module("quantization", GENERAL_SCRIPT)
E = load_module("e_quantization", E_SCRIPT)
DATASET = load_module("calibration_dataset_under_test", DATASET_SCRIPT)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def write_config(work_dir, **overrides):
    config = {
        "model_id": "org/model",
        "device_map": "cpu",
        "save_dir": "output",
        "use_bundled_chat_template": False,
        **overrides,
    }
    (work_dir / "config.yaml").write_text(
        json.dumps(config),
        encoding="utf-8",
    )


def quantization_contract(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    def call_named(name):
        return next(
            call
            for call in calls
            if (isinstance(call.func, ast.Name) and call.func.id == name)
            or (isinstance(call.func, ast.Attribute) and call.func.attr == name)
        )

    assignments = {}
    attributes = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            assignments[target.id] = node.value
        elif isinstance(target, ast.Attribute):
            attributes[ast.unparse(target)] = node.value

    model_load = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "from_pretrained"
        and ast.unparse(call.func.value) == "AutoModelForCausalLM"
    )
    recipe = assignments["recipe"]
    modifier = recipe.elts[0]
    modifier_keywords = {item.arg: item.value for item in modifier.keywords}
    kv_cache = modifier_keywords["kv_cache_scheme"]
    oneshot = call_named("oneshot")
    model_save = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and ast.unparse(call.func.value) == "model"
        and call.func.attr == "save_pretrained"
    )

    imatrix_gatherer_present = any(
        (
            isinstance(node, ast.ImportFrom)
            and node.module == "llmcompressor.modifiers.transform.imatrix"
        )
        or (
            isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == "IMatrixGatherer"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "IMatrixGatherer"
            )
        )
        for node in ast.walk(tree)
    )

    return {
        "dataset_import": [
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "calibration_dataset"
            for alias in node.names
        ],
        "model_args": [ast.unparse(argument) for argument in model_load.args],
        "model_kwargs": {
            item.arg: ast.unparse(item.value) for item in model_load.keywords
        },
        "ignore": ast.literal_eval(assignments["ignore"]),
        "preset": [
            ast.literal_eval(argument)
            for argument in call_named("preset_name_to_scheme").args
        ],
        "observer": ast.literal_eval(
            attributes["nvfp4_scheme.weights.observer"]
        ),
        "observer_kwargs": ast.literal_eval(
            attributes["nvfp4_scheme.weights.observer_kwargs"]
        ),
        "imatrix_gatherer_present": imatrix_gatherer_present,
        "recipe_modifiers": [ast.unparse(item.func) for item in recipe.elts],
        "config_groups": ast.unparse(modifier_keywords["config_groups"]),
        "modifier_ignore": ast.unparse(modifier_keywords["ignore"]),
        "kv_cache": {
            item.arg: ast.literal_eval(item.value) for item in kv_cache.keywords
        },
        "oneshot": {
            item.arg: ast.unparse(item.value) for item in oneshot.keywords
        },
        "save_compressed": ast.literal_eval(
            next(
                item.value
                for item in model_save.keywords
                if item.arg == "save_compressed"
            )
        ),
        "pipeline_symbols": {
            name for name in assignments if name.upper() == "PIPELINE" or name == "PIPELINES"
        },
    }


class Tests(unittest.TestCase):
    def test_config_and_template_selection(self):
        class Tokenizer:
            chat_template = "default"

        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            device_map = {"model.embed_tokens": "cpu", "model.layers.0": 0}
            write_config(
                work_dir,
                device_map=device_map,
                pipeline="ignored",
            )
            expected = ("org/model", device_map, work_dir / "output", False)
            for module in (GENERAL, E):
                with self.subTest(module=module.__name__):
                    self.assertEqual(module.load_config(work_dir), expected)

            write_config(work_dir, use_bundled_chat_template="true")
            for module in (GENERAL, E):
                with self.subTest(module=module.__name__):
                    with self.assertRaises(TypeError):
                        module.load_config(work_dir)

            for module in (GENERAL, E):
                with self.subTest(module=module.__name__):
                    tokenizer = Tokenizer()
                    template = work_dir / "chat_template.jinja"
                    template.unlink(missing_ok=True)
                    module.apply_chat_template_file(tokenizer, work_dir, False)
                    self.assertEqual(tokenizer.chat_template, "default")
                    with self.assertRaises(FileNotFoundError):
                        module.apply_chat_template_file(tokenizer, work_dir, True)
                    template.write_text("official", encoding="utf-8")
                    module.apply_chat_template_file(tokenizer, work_dir, True)
                    self.assertEqual(tokenizer.chat_template, "official")

    def test_invalid_tool_data_is_filtered(self):
        conversations = [
            [
                {
                    "from": "gpt",
                    "value": (
                        '<tool_call>{"id":"shared","name":"expected",'
                        '"arguments":{}}</tool_call>'
                    ),
                },
                {
                    "from": "tool",
                    "value": (
                        '<tool_response>{"tool_call_id":"shared","name":"wrong",'
                        '"content":{"ok":true}}</tool_response>'
                    ),
                },
            ],
            [
                {
                    "from": "gpt",
                    "value": (
                        '<tool_call>{"name":"first","arguments":{}}</tool_call>'
                        '<tool_call>{"name":"second","arguments":{}}</tool_call>'
                    ),
                },
                {"from": "tool", "value": '{"content":{"ok":true}}'},
            ],
            [
                {
                    "from": "gpt",
                    "value": '<tool_call>{"name":"overview","arguments":{}}</tool_call>',
                },
                {
                    "from": "tool",
                    "value": (
                        '<tool_response>{"name":"overview","content":"'
                        "{'album': \"Quoted title\"}"
                        '"}</tool_response>'
                    ),
                },
            ],
            [
                {
                    "from": "gpt",
                    "value": '<tool_call>{"name":"broken","arguments":[}</tool_call>',
                }
            ],
        ]
        for index, conversation in enumerate(conversations):
            with self.subTest(index=index):
                self.assertFalse(DATASET.has_consistent_tool_data(conversation))
                with self.assertRaises(DATASET.InvalidToolDataError):
                    DATASET.normalize_conversation(
                        conversation,
                        strict_tool_data=True,
                    )

    def test_hermes_sampling_preserves_semantics(self):
        selected = DATASET.select_hermes_samples(
            load_dataset(DATASET.HERMES_DATASET, split="train")
        )
        counts = {}
        for category in selected["scenario_category"]:
            counts[category] = counts.get(category, 0) + 1
        self.assertEqual(
            counts,
            {"single": 102, "multiturn": 154, "multistep": 205, "relevance": 51},
        )

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
        GENERAL.apply_chat_template_file(tokenizer, ROOT, True)

        for index, row in enumerate(selected):
            with self.subTest(index=index, category=row["scenario_category"]):
                messages = DATASET.normalize_conversation(
                    row["conversations"], strict_tool_data=True
                )
                actual_responses = [
                    response
                    for message in messages
                    for response in message.get("tool_responses", [])
                ]
                actual_payloads = {
                    canonical(response["response"])
                    for response in actual_responses
                }

                for turn in row["conversations"]:
                    visible = DATASET.THINK.sub("", str(turn.get("value", "")))
                    raw_responses = DATASET.TOOL_RESPONSE.findall(visible)
                    if turn["from"] == "tool" and not raw_responses:
                        raw_responses = [visible]
                    for raw_response in raw_responses:
                        response = DATASET.parse_response(raw_response)
                        if turn["from"] == "tool" or response.get(
                            "name"
                        ) or response.get("tool_call_id"):
                            self.assertIn(
                                canonical(response.get("content")),
                                actual_payloads,
                            )

                rendered = tokenizer.apply_chat_template(
                    messages,
                    tools=DATASET.normalize_tools(row["tools"]),
                    tokenize=False,
                    add_generation_prompt=False,
                    preserve_thinking=True,
                )
                self.assertEqual(
                    rendered.count("<|tool_response>response:"),
                    len(actual_responses),
                )
                for response in actual_responses:
                    self.assertIn(
                        f'<|tool_response>response:{response["name"]}',
                        rendered,
                    )
                self.assertLessEqual(
                    len(tokenizer.encode(rendered, add_special_tokens=False)),
                    8192,
                )

        calibration_dataset = DATASET.create_calibration_dataset(tokenizer)
        self.assertEqual(len(calibration_dataset), 1024)
        self.assertEqual(calibration_dataset.column_names, ["text"])
        self.assertTrue(all(calibration_dataset["text"]))

    def test_normalization_contract(self):
        tools = DATASET.normalize_tools(
            '[{"name":"search","parameters":{"count":{"type":"int"},'
            '"queries":{"type":"List[str]"}}}]'
        )
        properties = tools[0]["function"]["parameters"]["properties"]
        self.assertEqual(properties["count"]["type"], "integer")
        self.assertEqual(
            properties["queries"],
            {"type": "array", "items": {"type": "string"}},
        )

        messages = DATASET.normalize_conversation(
            [
                {
                    "from": "system",
                    "value": (
                        "Keep this.\nYou are a function calling AI model. "
                        "<tools>[]</tools>"
                    ),
                },
                {"from": "human", "value": "Check Tokyo and Osaka"},
                {
                    "from": "gpt",
                    "value": (
                        "<think>Mention `<tool_call>` only as text.</think>"
                        '<tool_call>{"name":"weather","arguments":{"city":"Tokyo"}}'
                        "</tool_call>"
                        '<tool_call>{"name":"weather","arguments":{"city":"Osaka"}}'
                        "</tool_call>"
                    ),
                },
                {
                    "from": "tool",
                    "value": (
                        '<tool_response>{"tool_call_id":"tokyo","name":"weather",'
                        '"content":{"temperature":28}}</tool_response>'
                        '<tool_response>{"tool_call_id":"osaka","name":"weather",'
                        '"content":{"temperature":29}}</tool_response>'
                    ),
                },
                {
                    "from": "gpt",
                    "value": (
                        '<tool_response>**{"tool_call_id":"duplicate","name":"weather",'
                        '"content":{"temperature":29}}**</tool_response>'
                    ),
                },
            ]
        )
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "assistant"],
        )
        self.assertEqual(messages[0]["content"], "Keep this.")
        self.assertEqual(
            messages[2]["reasoning_content"],
            "Mention `<tool_call>` only as text.",
        )
        self.assertEqual(
            messages[2]["tool_calls"],
            [
                {
                    "id": "tokyo",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": {"city": "Tokyo"},
                    },
                },
                {
                    "id": "osaka",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": {"city": "Osaka"},
                    },
                },
            ],
        )
        self.assertEqual(
            messages[2]["tool_responses"],
            [
                {"name": "weather", "response": {"temperature": 28}},
                {"name": "weather", "response": {"temperature": 29}},
            ],
        )
        self.assertEqual(messages[3]["content"], "")

        unwrapped = DATASET.normalize_conversation(
            [
                {
                    "from": "gpt",
                    "value": '<tool_call>{"name":"restaurant","arguments":{}}</tool_call>',
                },
                {
                    "from": "tool",
                    "value": "<tool_response>{'name':'Bistro','distance':500}</tool_response>",
                },
            ]
        )
        self.assertEqual(
            unwrapped[0]["tool_responses"],
            [
                {
                    "name": "restaurant",
                    "response": {"name": "Bistro", "distance": 500},
                }
            ],
        )

    def test_quantization_contract(self):
        general = quantization_contract(GENERAL_SCRIPT)
        e_model = quantization_contract(E_SCRIPT)

        for contract in (general, e_model):
            self.assertEqual(contract["dataset_import"], ["create_calibration_dataset"])
            self.assertEqual(contract["model_args"], ["model_id"])
            self.assertEqual(
                contract["model_kwargs"],
                {"dtype": "'auto'", "device_map": "device_map"},
            )
            self.assertEqual(contract["preset"], ["NVFP4", ["Linear"]])
            self.assertEqual(contract["observer"], "imatrix_mse")
            self.assertEqual(contract["observer_kwargs"], {"strict": True})
            self.assertFalse(contract["imatrix_gatherer_present"])
            self.assertEqual(contract["recipe_modifiers"], ["GPTQModifier"])
            self.assertEqual(
                contract["config_groups"],
                "{'group_0': nvfp4_scheme}",
            )
            self.assertEqual(contract["modifier_ignore"], "ignore")
            self.assertEqual(
                contract["kv_cache"],
                {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "tensor",
                    "dynamic": False,
                    "observer": "static_minmax",
                },
            )
            self.assertTrue(contract["save_compressed"])
            self.assertFalse(contract["pipeline_symbols"])

        self.assertEqual(
            general["ignore"],
            [
                "lm_head",
                "re:.*embed.*",
                "re:.*vision.*",
                "re:.*audio.*",
                "re:.*router.*",
            ],
        )
        self.assertEqual(
            e_model["ignore"],
            general["ignore"] + ["re:.*per_layer.*"],
        )

        oneshot = {
            "model": "model",
            "tokenizer": "tokenizer",
            "recipe": "recipe",
            "dataset": "calibration_dataset",
            "num_calibration_samples": "len(calibration_dataset)",
            "shuffle_calibration_samples": "False",
            "text_column": "'text'",
            "max_seq_length": "8192",
            "pad_to_max_length": "False",
        }
        self.assertEqual(general["oneshot"], oneshot)
        self.assertEqual(
            e_model["oneshot"],
            {**oneshot, "pipeline": "'basic'"},
        )


if __name__ == "__main__":
    unittest.main()
