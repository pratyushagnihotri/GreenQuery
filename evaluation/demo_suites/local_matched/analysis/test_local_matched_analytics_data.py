import ast
import base64
import gzip
import json
import unittest
from pathlib import Path

from local_matched_analytics_data import load_dataset


class PresentationDataTest(unittest.TestCase):
    def test_each_dataset_is_complete_and_cold_cache(self):
        movie = load_dataset("movie")
        rotowire = load_dataset("rotowire")
        medical = load_dataset("medical")
        self.assertEqual(len(movie), 80)
        self.assertEqual(len(rotowire), 40)
        self.assertEqual(len(medical), 80)
        self.assertEqual(movie.loc[movie["query_id"] == "Q1", "quality"].iat[0]["metric"], "precision_at_result_limit")
        qwen_q3 = movie.loc[(movie["query_id"] == "Q3") & (movie["model_label"] == "Qwen 2.5 7B"), "quality"].iat[0]
        self.assertEqual(qwen_q3["status"], "evaluated")
        self.assertEqual(qwen_q3["f1"], 0.0)
        self.assertTrue(all(item["status"] == "not_evaluated" for item in rotowire["quality"]))
        self.assertEqual(medical.loc[(medical["query_id"] == "Q4") & (medical["model_label"] == "Qwen 2.5 7B"), "quality"].iat[0]["absolute_error"], 1.25)

    def test_notebook_embeds_viewer_fallback_data(self):
        tree = ast.parse((Path(__file__).parent / "local_matched_analytics.py").read_text())
        cell = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_")
        assignment = next(node for node in cell.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_embedded_data" for target in node.targets))
        embedded = ast.literal_eval(assignment.value)
        self.assertEqual({name: len(json.loads(gzip.decompress(base64.b64decode(value)))) for name, value in embedded.items()}, {"movie": 80, "rotowire": 40, "medical": 80})


if __name__ == "__main__":
    unittest.main()
