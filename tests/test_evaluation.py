# test_model_performance.py
import pytest
import pandas as pd
import joblib
import os
from sklearn.metrics import roc_auc_score
from feature_engineering import create_features, simulate_student_interactions

# --- Configuration ---
MODELS_DIR = "models/"
LGBM_MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_psych_predictor_enriched.joblib")
SKILL_ENCODER_PATH = os.path.join(MODELS_DIR, "psych_skill_encoder.joblib")
GOLDEN_TEST_SET_PATH = "tests/golden_test_set.parquet"

MINIMUM_ACCEPTABLE_AUC = 0.75

@pytest.fixture(scope="module")
def models_and_data():
    """A single fixture to set up everything needed for the tests."""
    required_files = [LGBM_MODEL_PATH, SKILL_ENCODER_PATH, GOLDEN_TEST_SET_PATH]
    if not all(os.path.exists(p) for p in required_files):
        pytest.fail("FATAL: One or more required files are missing. Please run the notebook and create the golden test set.")
    
    lgbm_model = joblib.load(LGBM_MODEL_PATH)
    skill_encoder = joblib.load(SKILL_ENCODER_PATH)
    golden_df_static = pd.read_parquet(GOLDEN_TEST_SET_PATH)

    # Use the REAL simulation logic to create a test history
    golden_df_simulated = simulate_student_interactions(
        df_qa=golden_df_static, 
        num_students=50, 
        interactions_per_student=20
    )
    
    # Process the golden data using the EXACT same feature engineering function
    processed_golden_data = create_features(golden_df_simulated, skill_encoder)

    return {
        "lgbm": lgbm_model,
        "processed_data": processed_golden_data
    }

def test_lgbm_performance_threshold(models_and_data):
    """TRUST CHECK 1: Is the model's performance on a fixed dataset above our quality bar?"""
    model = models_and_data["lgbm"]
    test_df = models_and_data["processed_data"]
    
    if test_df.empty:
        pytest.fail("Processed golden test set is empty. Check feature engineering and data.")

    features = ['prior_response_time', 'prior_is_correct', 'skill_id_encoded', 'skill_attempts', 'skill_correct_rate', 'question_length']
    target = 'is_correct'
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    predictions = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, predictions)
    
    print(f"\nModel AUC on golden test set: {auc_score:.4f}")
    assert auc_score >= MINIMUM_ACCEPTABLE_AUC, f"Model performance has dropped below the threshold! AUC: {auc_score:.4f} < {MINIMUM_ACCEPTABLE_AUC}"

def test_model_feature_importance_is_stable(models_and_data):
    """TRUST CHECK 2: Does the model still 'think' the same way?"""
    model = models_and_data["lgbm"]
    
    feature_names = model.feature_name_
    importances = model.feature_importances_
    
    importance_dict = dict(zip(feature_names, importances))
    most_important_feature = max(importance_dict, key=importance_dict.get)
    
    print(f"\nMost important feature found: '{most_important_feature}'")
    assert most_important_feature == 'skill_correct_rate', "The model's most important feature has changed unexpectedly from 'skill_correct_rate'."