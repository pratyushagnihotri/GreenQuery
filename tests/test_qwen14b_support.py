import unittest
from pathlib import Path

from reasondb.backends.text_qa import PORT_KV_TEXT_QA


MODEL = "Qwen/Qwen2.5-14B-Instruct"
ROOT = Path(__file__).parents[1]


class Qwen14BSupportTests(unittest.TestCase):
    def test_model_is_registered_and_every_launcher_accepts_it(self):
        self.assertEqual(PORT_KV_TEXT_QA[MODEL], 5014)
        for name in ("movie", "rotowire", "medical", "artwork", "animals"):
            launcher = ROOT / "evaluation" / "scripts" / f"run_{name}_demo_repeated.sbatch"
            source = launcher.read_text()
            self.assertIn(MODEL, source)
            self.assertIn("DIRECT_QA_BATCH_SIZE=4", source)


if __name__ == "__main__":
    unittest.main()
