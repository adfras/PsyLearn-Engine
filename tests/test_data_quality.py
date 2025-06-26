# tests/test_data_quality.py
import pytest
import pandas as pd
from pathlib import Path

# --- Configuration Synchronized with prepare_data.py ---
PROCESSED_DATA_FILE = Path("data/2_processed_data/normalized_questions.parquet")

# The columns we ACTUALLY generate.
# Note: 'licence' is not part of the current pipeline, so it is not tested for.
EXPECTED_COLUMNS = ['question', 'answer', 'source']

# The data sources we ACTUALLY process. This ensures no unexpected data has been added.
EXPECTED_SOURCES = [
    'gsm8k',
    'pubmedqa',
    'boltmonkey',
    'mentat',
    'gragroo',
    'proofwriter',
    'thoughtsource',
]

MIN_QUESTION_LENGTH = 10
MAX_QUESTION_LENGTH = 1500 # A great check to prevent overly long content.

@pytest.fixture(scope="module")
def data():
    """A pytest fixture to load the main dataset once for all tests."""
    if not PROCESSED_DATA_FILE.exists():
        pytest.fail(f"FATAL: Processed data file not found at {PROCESSED_DATA_FILE}.")
    return pd.read_parquet(PROCESSED_DATA_FILE)

# --- Test Cases ---

def test_file_exists():
    """Test 1: Ensures the processed data file was actually created."""
    assert PROCESSED_DATA_FILE.exists(), "The final processed parquet file is missing."

def test_schema_is_correct(data):
    """Test 2: Validates that all expected columns are present."""
    actual_columns = set(data.columns)
    for col in EXPECTED_COLUMNS:
        assert col in actual_columns, f"Missing expected column: '{col}'"

def test_no_missing_critical_data(data):
    """Test 3: Ensures there are no nulls in the core 'question' and 'answer' fields."""
    assert data['question'].isnull().sum() == 0, "There are missing values in the 'question' column."
    assert data['answer'].isnull().sum() == 0, "There are missing values in the 'answer' column."

def test_content_plausibility(data):
    """Test 4: Checks if the data content is reasonable (e.g., not too short or long)."""
    shortest_question = data['question'].str.len().min()
    assert shortest_question >= MIN_QUESTION_LENGTH, f"Found a question with length {shortest_question}, which is shorter than the minimum threshold of {MIN_QUESTION_LENGTH}."
    
    longest_question = data['question'].str.len().max()
    assert longest_question <= MAX_QUESTION_LENGTH, f"Found a question with length {longest_question}, which is longer than the maximum threshold of {MAX_QUESTION_LENGTH}."

def test_source_column_is_valid(data):
    """
    Test 5: Checks if the 'source' column contains only known, expected values.
    This is crucial for ensuring data provenance is tracked correctly.
    """
    actual_sources = set(data['source'].unique())
    
    # Find any sources that are in our data but NOT in our official expected list.
    unexpected_sources = actual_sources - set(EXPECTED_SOURCES)
    
    assert not unexpected_sources, f"Found unexpected data sources: {unexpected_sources}. Please update the EXPECTED_SOURCES list in the test if this is intentional."