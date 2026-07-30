import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from datasets import load_dataset
from jinja2 import Environment
from transformers import AutoTokenizer

ROOT = Path(__file__).parent
SCRIPT = ROOT / "gemma4-nvfp4-quantization.py"
TEMPLATE = ROOT / "chat_template.jinja"
SPEC = importlib.util.spec_from_file_location("quantization", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class Tests(unittest.TestCase):
    def test_config_and_template_selection(self):
        class Tokenizer:
            chat_template = "default"

        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            (work_dir / "config.yaml").write_text(
                "model_id: org/model\ndevice_map: cpu\nsave_dir: output\n"
                "use_bundled_chat_template: false\n",
                encoding="utf-8",
            )
            model_id, device_map, save_dir, use_bundled_template = MODULE.load_config(work_dir)
            self.assertEqual(
                (model_id, device_map, save_dir, use_bundled_template),
                ("org/model", "cpu", work_dir / "output", False),
            )

            tokenizer = Tokenizer()
            template_path = work_dir / "chat_template.jinja"
            template_path.write_text("official", encoding="utf-8")
            MODULE.apply_chat_template_file(tokenizer, work_dir, use_bundled_template)
            self.assertEqual(tokenizer.chat_template, "default")

            template_path.unlink()
            with self.assertRaises(FileNotFoundError):
                MODULE.apply_chat_template_file(tokenizer, work_dir, True)

            template_path.write_text("official", encoding="utf-8")
            MODULE.apply_chat_template_file(tokenizer, work_dir, True)
            self.assertEqual(tokenizer.chat_template, "official")

            (work_dir / "config.yaml").write_text(
                "model_id: org/model\ndevice_map:\n  model.embed_tokens: cpu\n"
                "  model.layers.0: 0\nsave_dir: output\n"
                "use_bundled_chat_template: false\n",
                encoding="utf-8",
            )
            _, device_map, _, _ = MODULE.load_config(work_dir)
            self.assertEqual(
                device_map,
                {"model.embed_tokens": "cpu", "model.layers.0": 0},
            )

            (work_dir / "config.yaml").write_text(
                'model_id: org/model\ndevice_map: auto\nsave_dir: output\n'
                'use_bundled_chat_template: "true"\n',
                encoding="utf-8",
            )
            with self.assertRaises(TypeError):
                MODULE.load_config(work_dir)

    def test_hermes_sampling(self):
        mismatched = [
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
        ]
        self.assertFalse(MODULE.has_consistent_tool_data(mismatched))
        with self.assertRaises(MODULE.UnmatchedToolResponseError):
            MODULE.normalize_conversation(
                mismatched, strict_tool_data=True
            )

        ambiguous = [
            {
                "from": "gpt",
                "value": (
                    '<tool_call>{"name":"first","arguments":{}}</tool_call>'
                    '<tool_call>{"name":"second","arguments":{}}</tool_call>'
                ),
            },
            {"from": "tool", "value": '{"content":{"ok":true}}'},
        ]
        self.assertFalse(MODULE.has_consistent_tool_data(ambiguous))

        self.assertEqual(
            MODULE.HERMES_SAMPLES,
            {"single": 102, "multiturn": 154, "multistep": 205, "relevance": 51},
        )
        selected = MODULE.select_hermes_samples(
            load_dataset(MODULE.HERMES_DATASET, split="train")
        )

        counts = {}
        for category in selected["scenario_category"]:
            counts[category] = counts.get(category, 0) + 1
        self.assertEqual(counts, MODULE.HERMES_SAMPLES)
        self.assertEqual(len(selected), 512)

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
        MODULE.apply_chat_template_file(tokenizer, ROOT, True)

        for index, row in enumerate(selected):
            with self.subTest(index=index, category=row["scenario_category"]):
                self.assertTrue(
                    MODULE.has_consistent_tool_data(row["conversations"])
                )
                messages = MODULE.normalize_conversation(
                    row["conversations"], strict_tool_data=True
                )
                tools = MODULE.normalize_tools(row["tools"])

                expected_responses = []
                for turn in row["conversations"]:
                    visible = MODULE.THINK.sub("", str(turn.get("value", "")))
                    raw_responses = MODULE.TOOL_RESPONSE.findall(visible)
                    if turn["from"] == "tool" and not raw_responses:
                        raw_responses = [visible]
                    for raw_response in raw_responses:
                        response = MODULE.parse_response(raw_response)
                        if turn["from"] == "tool" or response.get(
                            "name"
                        ) or response.get("tool_call_id"):
                            expected_responses.append(
                                canonical(response.get("content"))
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
                for expected in expected_responses:
                    self.assertIn(expected, actual_payloads)

                rendered = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                    preserve_thinking=True,
                )
                self.assertTrue(rendered)
                self.assertEqual(
                    rendered.count("<|tool_response>response:"), len(actual_responses)
                )
                for response in actual_responses:
                    self.assertIn(
                        f'<|tool_response>response:{response["name"]}', rendered
                    )
                self.assertLessEqual(
                    len(tokenizer.encode(rendered, add_special_tokens=False)),
                    8192,
                )

    def test_tool_schema_normalization(self):
        tools = MODULE.normalize_tools(
            json.dumps(
                [
                    {
                        "name": "search",
                        "parameters": {
                            "count": {"type": "int"},
                            "queries": {"type": "List[str]"},
                        },
                    }
                ]
            )
        )
        properties = tools[0]["function"]["parameters"]["properties"]
        self.assertEqual(properties["count"]["type"], "integer")
        self.assertEqual(
            properties["queries"], {"type": "array", "items": {"type": "string"}}
        )

    def test_conversation_normalization(self):
        messages = MODULE.normalize_conversation(
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
        self.assertEqual([message["role"] for message in messages], [
            "system", "user", "assistant", "assistant"
        ])
        self.assertEqual(messages[0]["content"], "Keep this.")
        self.assertEqual(len(messages[2]["tool_calls"]), 2)
        self.assertEqual(
            messages[2]["tool_responses"],
            [
                {"name": "weather", "response": {"temperature": 28}},
                {"name": "weather", "response": {"temperature": 29}},
            ],
        )
        self.assertEqual(messages[3]["content"], "")

    def test_response_edge_cases(self):
        unwrapped = MODULE.normalize_conversation(
            [
                {
                    "from": "gpt",
                    "value": '<tool_call>{"name":"restaurant","arguments":{}}</tool_call>',
                },
                {
                    "from": "tool",
                    "value": (
                        "<tool_response>{'name':'Bistro','distance':500}</tool_response>"
                        "<tool_response>{'unrelated':true}</tool_response>"
                    ),
                },
            ]
        )
        self.assertEqual(
            unwrapped[0]["tool_responses"],
            [{"name": "restaurant", "response": {"name": "Bistro", "distance": 500}}],
        )

        malformed_conversation = [
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
        ]
        malformed = MODULE.normalize_conversation(malformed_conversation)
        self.assertEqual(malformed[0]["tool_responses"][0]["name"], "overview")
        self.assertIsInstance(malformed[0]["tool_responses"][0]["response"], str)
        self.assertFalse(MODULE.has_consistent_tool_data(malformed_conversation))

        orphan = MODULE.normalize_conversation(
            [{"from": "gpt", "value": "<tool_response>Natural answer</tool_response>"}]
        )
        self.assertEqual(orphan, [{"role": "assistant", "content": "Natural answer"}])

        invalid_call_conversation = [
            {
                "from": "gpt",
                "value": '<tool_call>{"name":"broken","arguments":[}</tool_call>',
            }
        ]
        invalid_call = MODULE.normalize_conversation(invalid_call_conversation)
        self.assertEqual(
            invalid_call,
            [{"role": "assistant", "content": '{"name":"broken","arguments":[}'}],
        )
        self.assertFalse(MODULE.has_consistent_tool_data(invalid_call_conversation))

    def test_official_template_renders_normalized_data(self):
        messages = MODULE.normalize_conversation(
            [
                {
                    "from": "gpt",
                    "value": '<tool_call>{"name":"weather","arguments":{}}</tool_call>',
                },
                {
                    "from": "tool",
                    "value": (
                        '<tool_response>{"name":"weather","content":'
                        '{"data":[{"temperature":28}]}}</tool_response>'
                    ),
                },
            ]
        )
        rendered = Environment().from_string(TEMPLATE.read_text(encoding="utf-8")).render(
            messages=messages,
            tools=MODULE.normalize_tools('[{"name":"weather","parameters":{}}]'),
            bos_token="<bos>",
            add_generation_prompt=False,
            preserve_thinking=True,
            enable_thinking=False,
        )
        self.assertIn("<|tool_call>call:weather", rendered)
        self.assertIn("<|tool_response>response:weather{data:[{temperature:28}]}", rendered)

    def test_quantization_contract(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

        def call_named(name):
            return next(
                call
                for call in calls
                if (isinstance(call.func, ast.Name) and call.func.id == name)
                or (isinstance(call.func, ast.Attribute) and call.func.attr == name)
            )

        model_load = next(
            call
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and ast.unparse(call.func.value) == "AutoModelForCausalLM"
            and call.func.attr == "from_pretrained"
        )
        model_keywords = {item.arg: item.value for item in model_load.keywords}
        self.assertEqual(model_keywords["dtype"].value, "auto")
        self.assertEqual(ast.unparse(model_keywords["device_map"]), "device_map")

        template_calls = [
            call
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "apply_chat_template"
        ]
        self.assertEqual(len(template_calls), 2)
        self.assertTrue(all(
            {item.arg: item.value for item in call.keywords}["tokenize"].value is False
            for call in template_calls
        ))

        assignments = {
            target.id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance((target := node.targets[0]), ast.Name)
        }
        self.assertEqual(
            ast.literal_eval(assignments["ignore"]),
            [
                "lm_head",
                "re:.*embed.*",
                "re:.*vision.*",
                "re:.*audio.*",
                "re:.*router.*",
                "re:.*per_layer.*",
            ],
        )

        preset = call_named("preset_name_to_scheme")
        self.assertEqual(ast.literal_eval(preset.args[0]), "NVFP4")
        self.assertEqual(ast.literal_eval(preset.args[1]), ["Linear"])

        attribute_assignments = {
            ast.unparse(node.targets[0]): node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
        }
        self.assertEqual(
            ast.literal_eval(attribute_assignments["nvfp4_scheme.weights.observer"]),
            "imatrix_mse",
        )
        self.assertEqual(
            ast.literal_eval(
                attribute_assignments["nvfp4_scheme.weights.observer_kwargs"]
            ),
            {"strict": True},
        )

        self.assertFalse(
            any(
                isinstance(node, ast.ImportFrom)
                and node.module == "llmcompressor.modifiers.transform.imatrix"
                for node in ast.walk(tree)
            )
        )

        recipe = assignments["recipe"]
        self.assertIsInstance(recipe, ast.List)
        self.assertEqual(
            [ast.unparse(item.func) for item in recipe.elts],
            ["GPTQModifier"],
        )

        modifier_keywords = {
            item.arg: item.value for item in recipe.elts[0].keywords
        }
        config_groups = modifier_keywords["config_groups"]
        self.assertEqual(
            [ast.literal_eval(key) for key in config_groups.keys], ["group_0"]
        )
        self.assertEqual(
            [ast.unparse(value) for value in config_groups.values], ["nvfp4_scheme"]
        )
        self.assertEqual(ast.unparse(modifier_keywords["ignore"]), "ignore")

        kv_cache_scheme = modifier_keywords["kv_cache_scheme"]
        self.assertEqual(ast.unparse(kv_cache_scheme.func), "QuantizationArgs")
        self.assertEqual(
            {
                item.arg: ast.literal_eval(item.value)
                for item in kv_cache_scheme.keywords
            },
            {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "tensor",
                "dynamic": False,
                "observer": "static_minmax",
            },
        )

        oneshot = call_named("oneshot")
        oneshot_keywords = {item.arg: item.value for item in oneshot.keywords}
        self.assertEqual(ast.unparse(oneshot_keywords["tokenizer"]), "tokenizer")
        self.assertEqual(oneshot_keywords["text_column"].value, "text")
        self.assertEqual(oneshot_keywords["max_seq_length"].value, 8192)
        self.assertIs(oneshot_keywords["shuffle_calibration_samples"].value, False)
        self.assertIs(oneshot_keywords["pad_to_max_length"].value, False)
        self.assertNotIn("data_collator", oneshot_keywords)

        model_save = next(
            call
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and ast.unparse(call.func.value) == "model"
            and call.func.attr == "save_pretrained"
        )
        self.assertIs(
            {item.arg: item.value for item in model_save.keywords}[
                "save_compressed"
            ].value,
            True,
        )


if __name__ == "__main__":
    unittest.main()
