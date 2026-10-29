import pandas as pd
import numpy as np


class MetricsManager:
    def __init__(self, predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame):
        """
        Initializes using fully vectorized Pandas and NumPy operations.
        """
        self.preds_df = self._preprocess(predictions_df)
        self.gt_df = self._preprocess(ground_truth_df)

    def _preprocess(self, df: pd.DataFrame):
        # 1. Vectorized String Conversion & Cleaning
        # Converts all to strings, strips whitespace, and lowercases everything
        clean_df = df.astype(str)
        for col in clean_df.columns:
            clean_df[col] = clean_df[col].str.strip().str.lower()

        # 2. Targeted Vectorized Replacement
        # Safely extracts integers from pure float strings (e.g., "1.0" -> "1", "-42.00" -> "-42")
        # while strictly ignoring text strings (e.g., "Version 2.0" remains unchanged)
        clean_df = clean_df.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)

        # 3. Simulate `frozenset` (Column Order Independence)
        # We sort each row internally using NumPy, which is incredibly fast
        sorted_arr = np.sort(clean_df.values, axis=1)

        # 4. Simulate `set` (Row Uniqueness)
        # Rebuild the DataFrame and drop duplicate rows
        return pd.DataFrame(sorted_arr).drop_duplicates()

    def _compute_confusion_counts(self):
        # Vectorized Intersection
        # An inner merge finds exactly the rows that exist in both DataFrames
        tp_df = pd.merge(self.preds_df, self.gt_df, how="inner")

        TP = len(tp_df)
        FP = len(self.preds_df) - TP
        FN = len(self.gt_df) - TP
        return TP, FP, FN

    def compute_precision(self):
        TP, FP, _ = self._compute_confusion_counts()
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0

    def compute_recall(self):
        TP, _, FN = self._compute_confusion_counts()
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    def compute_f1(self, precision, recall):
        return (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

    def evaluate_all(self):
        """Compute all metrics and return them as a dictionary."""
        precision = self.compute_precision()
        recall = self.compute_recall()
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": self.compute_f1(precision, recall),
        }
