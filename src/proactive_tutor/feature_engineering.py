# feature_engineering.py

import pandas as pd
import numpy as np
import joblib
import os
import re
from pyBKT.models import Model

def simulate_student_interactions(df_qa, num_students, interactions_per_student):
    """Generates a realistic, time-series log from static Q&A data."""
    if df_qa is None or df_qa.empty:
        return pd.DataFrame()

    print(f"\nSimulating interaction logs for {num_students} students...")
    all_interactions = []

    for student_id in range(num_students):
        student_interactions = df_qa.sample(n=interactions_per_student, replace=True).copy()
        student_interactions['student_id'] = student_id

        mastery = {source: 0.1 for source in df_qa['source'].unique()}
        correct_list = []
        for _, row in student_interactions.iterrows():
            source = row['source']
            is_correct = 1 if np.random.rand() < mastery.get(source, 0.1) else 0
            correct_list.append(is_correct)
            mastery[source] += (1 - mastery.get(source, 0.1)) * 0.25 if is_correct else -mastery.get(source, 0.1) * 0.1
            mastery[source] = np.clip(mastery[source], 0.01, 0.99)

        student_interactions['is_correct'] = correct_list

        correct_times = np.random.normal(25, 5, size=len(student_interactions))
        incorrect_times = np.random.normal(60, 15, size=len(student_interactions))
        student_interactions['response_time_sec'] = np.where(student_interactions['is_correct'] == 1, correct_times, incorrect_times).clip(5, 300)

        student_interactions['timestamp'] = pd.to_datetime(pd.Timestamp.now() + pd.to_timedelta(np.arange(len(student_interactions)), 'm'))
        all_interactions.append(student_interactions)

    df_simulated = pd.concat(all_interactions, ignore_index=True) if all_interactions else pd.DataFrame()
    print(f"Simulation complete. Generated {len(df_simulated):,} interactions.")
    return df_simulated

# --- Helper function for BKT prior mastery calculation ---
def _calculate_bkt_mastery_for_history(student_skill_history_df, bkt_models, skill_name):
    """
    Calculates the prior mastery probability for a skill based on a student's history
    using the corresponding BKT model.
    """
    bkt_model = bkt_models.get(skill_name)
    if not bkt_model:
        return [0.5] * len(student_skill_history_df)

    bkt_data = student_skill_history_df.copy()
    bkt_data['user_id'] = 0 # Dummy user_id for this slice
    bkt_data['skill_name'] = skill_name
    bkt_data['correct'] = bkt_data['is_correct']
    bkt_data['order_id'] = np.arange(len(bkt_data)) + 1

    try:
        # --- THIS IS THE FIX ---
        # The .predict() method does NOT take a 'skills' argument.
        # It uses the skill_name column from the provided data.
        predictions = bkt_model.predict(data=bkt_data)
        # --- END OF FIX ---

        initial_prior = bkt_model.params().get(skill_name, [[0.1, 0.1, 0.2, 0.5]])[0][3]
        prior_masteries = [initial_prior] + predictions['state_predictions'].iloc[:-1].tolist()
        
        if len(prior_masteries) != len(student_skill_history_df):
             print(f"Warning: Mismatch in BKT prior calculation for skill {skill_name}. Expected {len(student_skill_history_df)}, got {len(prior_masteries)}. Using default prior.")
             return [0.5] * len(student_skill_history_df)

        return prior_masteries
    except Exception as e:
        print(f"Warning: BKT prediction failed for skill {skill_name} on student history: {e}. Using default prior.")
        return [0.5] * len(student_skill_history_df)


def create_features(df, skill_encoder, bkt_models=None):
    """
    Takes a dataframe of student interactions and engineers the features
    needed for the LGBM model.
    """
    processed_df = df.copy()

    known_sources = skill_encoder.classes_
    processed_df = processed_df[processed_df['source'].isin(known_sources)].copy()

    if processed_df.empty:
        return pd.DataFrame()

    processed_df['skill_id_encoded'] = skill_encoder.transform(processed_df['source'])
    processed_df.sort_values(['student_id', 'timestamp'], inplace=True, kind='mergesort')

    processed_df['prior_is_correct'] = processed_df.groupby('student_id')['is_correct'].shift(1)
    processed_df['prior_response_time'] = processed_df.groupby('student_id')['response_time_sec'].shift(1)
    processed_df['skill_attempts'] = processed_df.groupby(['student_id', 'skill_id_encoded']).cumcount()
    processed_df['skill_correct_sum_prev'] = processed_df.groupby(['student_id', 'skill_id_encoded'])['is_correct'].cumsum().shift(1).fillna(0)
    processed_df['skill_correct_rate'] = processed_df['skill_correct_sum_prev'] / processed_df['skill_attempts']
    processed_df['skill_correct_rate'] = processed_df['skill_correct_rate'].fillna(0.5)
    processed_df.loc[processed_df['skill_attempts'] == 0, 'skill_correct_rate'] = 0.5
    processed_df.drop(columns=['skill_correct_sum_prev'], inplace=True)

    processed_df['question_length'] = processed_df['question'].str.len().fillna(0)

    if bkt_models:
        print("Calculating BKT prior mastery for each interaction...")
        all_masteries = []
        # We need to process group by group to handle sequential nature of BKT
        grouped = processed_df.groupby(['student_id', 'skill_id_encoded'])
        for _, group_df in grouped:
            # We must get the original skill name to look up the correct BKT model
            skill_name = skill_encoder.inverse_transform([group_df['skill_id_encoded'].iloc[0]])[0]
            masteries = _calculate_bkt_mastery_for_history(group_df, bkt_models, skill_name)
            all_masteries.extend(masteries)
        
        # This assignment assumes the order of processed_df is preserved
        processed_df['bkt_prior_mastery'] = all_masteries
    else:
        print("BKT models not provided. 'bkt_prior_mastery' column will be filled with default values.")
        processed_df['bkt_prior_mastery'] = 0.5

    processed_df.dropna(subset=['prior_is_correct', 'prior_response_time'], inplace=True)
    
    return processed_df