# tests/test_augmented_data_quality.py

import pytest
import pandas as pd
from pathlib import Path

# --- Configuration (Should match your generation script) ---
AUGMENTED_DATA_FILE = Path("data/4_training_sets/distractor_generation_training_data_DOMAIN_AWARE_FIXED.parquet")
EXPECTED_COLUMNS = [
    'question',
    'correct_answer',
    'distractor',
    'question_source',
    'distractor_source'
]
PSYCHOLOGY_SOURCES = ['boltmonkey', 'gragroo', 'mentat']

# --- Pytest Fixture to Load Data Once ---
@pytest.fixture(scope="module")
def augmented_data():
    """Loads the generated parquet file once for all tests in this module."""
    if not AUGMENTED_DATA_FILE.exists():
        pytest.fail(f"FATAL: Augmented data file not found at {AUGMENTED_DATA_FILE}. Please run the generation script first.")
    
    df = pd.read_parquet(AUGMENTED_DATA_FILE)
    return df

# --- Test Cases ---

def test_file_creation_and_schema(augmented_data):
    """
    Test 1: Verifies that the file was created and has the exact columns we expect.
    """
    assert augmented_data is not None, "Dataframe failed to load."
    assert set(augmented_data.columns) == set(EXPECTED_COLUMNS), f"Column mismatch. Expected {EXPECTED_COLUMNS} but got {list(augmented_data.columns)}"

def test_no_missing_values(augmented_data):
    """
    Test 2: Ensures there are no null or empty values in any of the critical columns.
    """
    for col in EXPECTED_COLUMNS:
        assert augmented_data[col].isnull().sum() == 0, f"Found null values in column '{col}'."
        # Also check for empty strings which can be problematic
        if pd.api.types.is_string_dtype(augmented_data[col]):
            assert (augmented_data[col].str.len() > 0).all(), f"Found empty strings in column '{col}'."

def test_distractors_are_meaningful(augmented_data):
    """
    Test 3: A critical quality check to ensure a distractor is never the same as the correct answer.
    """
    num_identical_answers = (augmented_data['correct_answer'] == augmented_data['distractor']).sum()
    assert num_identical_answers == 0, f"Found {num_identical_answers} rows where the distractor is identical to the correct answer."

def test_domain_awareness_logic(augmented_data):
    """
    Test 4: The MOST IMPORTANT test. Verifies that the domain-aware logic was correctly applied.
    - Psychology questions should get psychology distractors.
    - Non-psychology questions should get distractors from their own source.
    """
    print("\n--- Verifying Domain-Aware Logic ---")

    # Part A: Test the psychology sources
    psych_df = augmented_data[augmented_data['question_source'].isin(PSYCHOLOGY_SOURCES)]
    if not psych_df.empty:
        print(f"Found {len(psych_df)} psychology-sourced questions. Checking their distractors...")
        are_distractors_psych = psych_df['distractor_source'].isin(PSYCHOLOGY_SOURCES)
        assert are_distractors_psych.all(), "Found psychology questions paired with non-psychology distractors."
        print("✅ Psychology question pairing is correct.")
    else:
        print("INFO: No psychology-sourced questions found in the sample to test.")

    # Part B: Test the non-psychology sources
    non_psych_df = augmented_data[~augmented_data['question_source'].isin(PSYCHOLOGY_SOURCES)]
    if not non_psych_df.empty:
        print(f"Found {len(non_psych_df)} non-psychology questions. Checking their distractors...")
        are_sources_matched = (non_psych_df['question_source'] == non_psych_df['distractor_source'])
        assert are_sources_matched.all(), "Found non-psychology questions where the distractor source does not match the question source."
        print("✅ Non-psychology question pairing is correct.")
    else:
        print("INFO: No non-psychology-sourced questions found in the sample to test.")