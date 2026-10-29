"""Quick test to verify row count extraction works."""

import sys
from pathlib import Path

# Add greenQueryLab to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from reasondb.interface.connect import RaccoonDB
from reasondb.database.indentifier import InPlaceColumn
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee

# Dataset path
csv = (
    Path(__file__).parents[2]
    / "reasondb"
    / "evaluation"
    / "benchmarks"
    / "files"
    / "reviews_100.csv"
)

print("🧪 Testing row count extraction...")
print(f"   Dataset: {csv}")
print()

with RaccoonDB("test_row_extraction", device="cpu") as rc:
    reviews = rc.add_table(
        path=csv,
        table_name="reviews",
        text_columns=[InPlaceColumn("reviews.reviewtext")],
    )

    # Q1: Simple query
    print("🔍 Running Query 1...")
    df_query = (
        reviews
        .filter("{reviewtext} is clearly positive")
        .project("Keep {reviewid}")
        .limit("Keep 5 rows")
    )
    result = df_query.execute(
        "test_positive_reviews",
        PrecisionGuarantee(0.8),
        RecallGuarantee(0.8),
    )

    print("✅ Query executed successfully")
    print()

    # Test different extraction methods
    print("📊 Testing extraction methods:")
    print()

    # Method 1: Check if result has to_df()
    if hasattr(result, 'to_df'):
        print("✓ Result has to_df() method")
        df = result.to_df()
        print(f"  Row count from to_df(): {len(df)}")
        print(f"  DataFrame shape: {df.shape}")
        print(f"  DataFrame columns: {list(df.columns)}")
    else:
        print("✗ Result does not have to_df() method")

    print()

    # Method 2: Check if result has __len__
    if hasattr(result, '__len__'):
        print(f"✓ Result has __len__: {len(result)}")
    else:
        print("✗ Result does not have __len__")

    print()

    # Method 3: Check result type and attributes
    print(f"Result type: {type(result)}")
    print(f"Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')][:10]}")

    print()
    print("🎯 Recommended extraction method:")
    print("   Use result.to_df() to get DataFrame, then len(df) for row count")

print()
print("✅ Test complete!")

# Made with Bob
