import asyncio
import importlib.util
import unittest
from unittest.mock import Mock, patch

DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(module) is not None
    for module in ("requests", "pandas", "torch", "duckdb")
)
if DEPENDENCIES_AVAILABLE:
    from reasondb.backends.audio_model import KvAudioModel
    from reasondb.backends.text_qa import KvTextQABackend
    from reasondb.reasoning.llm import local_planner_max_new_tokens


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "ReasonDB runtime dependencies unavailable")
class LocalInferenceTests(unittest.TestCase):
    def test_local_planner_generation_cap_is_validated(self):
        with patch.dict("os.environ", {"GREENQUERY_PLANNER_MAX_NEW_TOKENS": "128"}):
            self.assertEqual(local_planner_max_new_tokens(), 128)
        with patch.dict("os.environ", {"GREENQUERY_PLANNER_MAX_NEW_TOKENS": "0"}):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                local_planner_max_new_tokens()

    def test_disabled_cache_skips_cache_preparation(self):
        backend = KvTextQABackend("meta-llama/Llama-3.1-8B-Instruct", 0.0, False)
        with patch("reasondb.backends.text_qa.requests.post") as post:
            asyncio.run(backend.prepare(Mock(), Mock(), ["text"]))
        post.assert_not_called()

    def test_direct_response_requires_and_records_positive_tokens(self):
        backend = KvTextQABackend("meta-llama/Llama-3.1-8B-Instruct", 0.0, False)
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "answers": ["1"],
            "log_odds": [2.5],
            "telemetry": {
                "input_tokens": 20,
                "output_tokens": 1,
                "execution_mode": "direct_no_cache",
            },
        }
        with patch.dict("os.environ", {"GREENQUERY_DIRECT_QA_BATCH_SIZE": "8"}), patch(
            "reasondb.backends.text_qa.requests.post", return_value=response
        ) as post:
            rows, telemetry = backend._invoke_direct_no_cache(
                ["condition"], ["context"], True
            )
        self.assertEqual(post.call_args.kwargs["json"]["batch_size"], 8)
        self.assertEqual(rows[0][0], "1")
        self.assertGreater(telemetry["input_tokens"], 0)
        self.assertGreater(telemetry["output_tokens"], 0)
        self.assertEqual(telemetry["answer_cache_hits"], 0)
        self.assertEqual(telemetry["kv_cache_hits"], 0)
        self.assertEqual(telemetry["inference_batch_size"], 8)

    def test_direct_batch_size_rejects_zero(self):
        backend = KvTextQABackend("meta-llama/Llama-3.1-8B-Instruct", 0.0, False)
        with patch.dict("os.environ", {"GREENQUERY_DIRECT_QA_BATCH_SIZE": "0"}):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                backend._invoke_direct_no_cache(["condition"], ["context"], True)

    def test_disabled_audio_cache_skips_preparation(self):
        backend = KvAudioModel("Qwen/Qwen2-Audio-7B-Instruct", 0.0, False)
        with patch("reasondb.backends.audio_model.requests.post") as post:
            asyncio.run(backend.prepare(Mock(name="audio"), Mock(), [Mock()]))
        post.assert_not_called()

    def test_direct_audio_records_timestamped_telemetry(self):
        backend = KvAudioModel("Qwen/Qwen2-Audio-7B-Instruct", 0.0, False)
        response = Mock(status_code=200)
        response.json.return_value = {
            "answers": {"clip.wav": "1"}, "log_odds": {"clip.wav": 2.0},
            "telemetry": {"input_tokens": 20, "output_tokens": 1, "inference_started_at": 10.0, "inference_finished_at": 11.0},
        }
        with patch.dict("os.environ", {"GREENQUERY_AUDIO_QA_BATCH_SIZE": "8"}), patch(
            "reasondb.backends.audio_model.requests.post", return_value=response
        ) as post, patch("reasondb.backends.audio_model.record_local_inference") as record:
            asyncio.run(backend._invoke(Mock(name="audio"), "lion?", [__import__("pathlib").Path("clip.wav")], Mock(), True, Mock()))
        self.assertTrue(post.call_args.args[0].endswith("/audio_qa_direct"))
        self.assertEqual(record.call_args.args[0]["event_kind"], "audio")
        self.assertEqual(record.call_args.args[0]["inference_started_at"], 10.0)
        self.assertEqual(record.call_args.args[0]["inference_finished_at"], 11.0)


if __name__ == "__main__":
    unittest.main()
