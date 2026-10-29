import unittest

from evaluation.scripts.run_movie_measurement import movie_database_name


class MovieNamespaceTests(unittest.TestCase):
    def test_namespace_changes_the_reason_db_and_cold_cache_name(self):
        self.assertEqual(movie_database_name("demo", "qwen_123"), "movie_sampled_qwen_123")
        self.assertEqual(movie_database_name("semantic", "llama_456"), "movie_sampled_semantic_llama_456")


if __name__ == "__main__":
    unittest.main()
