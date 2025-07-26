# tests/test_data_quality.py
import pytest
pd = pytest.importorskip("pandas")
from pathlib import Path

PROCESSED_DATA_FILE = Path("data/2_processed_data/normalized_questions.parquet")
EXPECTED_COLUMNS = ["question", "answer", "source"]
EXPECTED_SOURCES = [
    "gsm8k",
    "pubmedqa",
    "boltmonkey",
    "mentat",
    "gragroo",
    "proofwriter",
    "thoughtsource",
    "train",
    "test",
    "validation",
    "bioasq",
]
MIN_QUESTION_LENGTH = 10
MAX_QUESTION_LENGTH = 5000

@pytest.fixture(scope="module")
def data():
    if not PROCESSED_DATA_FILE.exists():
        pytest.skip(f"Processed data file not found at {PROCESSED_DATA_FILE}.")
    return pd.read_parquet(PROCESSED_DATA_FILE)

def test_file_exists():
    if not PROCESSED_DATA_FILE.exists():
        pytest.skip(f"Processed data file not found at {PROCESSED_DATA_FILE}.")
    assert PROCESSED_DATA_FILE.exists()

def test_schema_is_correct(data):
    actual_columns = set(data.columns)
    for col in EXPECTED_COLUMNS:
        assert col in actual_columns

def test_no_missing_critical_data(data):
    assert data["question"].isnull().sum() == 0
    assert data["answer"].isnull().sum() == 0

def test_content_plausibility(data):
    shortest_question = data["question"].str.len().min()
    assert shortest_question >= MIN_QUESTION_LENGTH
    longest_question = data["question"].str.len().max()
    assert longest_question <= MAX_QUESTION_LENGTH

def test_source_column_is_valid(data):
    actual_sources = set(data["source"].unique())
    unexpected_sources = actual_sources - set(EXPECTED_SOURCES)
    assert not unexpected_sources
