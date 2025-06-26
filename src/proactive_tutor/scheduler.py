# scheduler.py
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os
import re

class SpacedRepetitionScheduler:
    INTERVALS = [1, 3, 7, 14, 30, 90, 180, 365] 
    BKT_MASTERY_THRESHOLD = 0.90

    def __init__(self, bkt_models_dir="models/", skill_encoder_path="models/psych_skill_encoder.joblib"):
        self.student_skill_states = {}
        self.bkt_models = self._load_bkt_models(bkt_models_dir)
        self.skill_encoder = joblib.load(skill_encoder_path) if os.path.exists(skill_encoder_path) else None
        
        if not self.skill_encoder:
            print("Warning: Skill encoder not found. Scheduler will operate on skill names directly.")

    def _load_bkt_models(self, models_dir):
        bkt_models = {}
        if not os.path.exists(models_dir):
            print(f"Warning: BKT models directory '{models_dir}' not found. Spaced repetition will be less accurate.")
            return bkt_models
        
        for filename in os.listdir(models_dir):
            if filename.startswith('bkt_model_') and filename.endswith('.pkl'):
                # We need to reconstruct the original skill name from the safe filename
                safe_skill_name = filename.replace('bkt_model_', '').replace('.pkl', '')
                
                # This logic assumes the only special character replaced was '/'
                # You may need to make this more robust if other characters were replaced
                skill_name_reconstructed = re.sub(r'_', '/', safe_skill_name)
                
                try:
                    # --- THIS IS THE FIX ---
                    # Use joblib.load() instead of Model.load()
                    bkt_models[skill_name_reconstructed] = joblib.load(os.path.join(models_dir, filename))
                    # --- END OF FIX ---
                except Exception as e:
                    print(f"Error loading BKT model {filename}: {e}")
        return bkt_models

    def update_skill_status(self, student_id: int, skill_name: str, is_correct: int, current_timestamp: datetime, bkt_mastery_after_interaction: float):
        if student_id not in self.student_skill_states:
            self.student_skill_states[student_id] = {}
        
        current_state = self.student_skill_states[student_id].get(skill_name, {'interval_idx': 0, 'last_practice_time': current_timestamp, 'bkt_mastery_at_last_update': 0.0})

        if is_correct == 1 and bkt_mastery_after_interaction >= self.BKT_MASTERY_THRESHOLD:
            current_state['interval_idx'] = min(current_state['interval_idx'] + 1, len(self.INTERVALS) - 1)
        elif is_correct == 0:
            current_state['interval_idx'] = 0

        current_state['last_practice_time'] = current_timestamp
        current_state['bkt_mastery_at_last_update'] = bkt_mastery_after_interaction
        self.student_skill_states[student_id][skill_name] = current_state

    def get_skills_due_for_review(self, student_id: int, current_timestamp: datetime) -> list:
        due_skills = []
        if student_id not in self.student_skill_states:
            return due_skills
        
        for skill_name, state in self.student_skill_states[student_id].items():
            last_practice_time = state['last_practice_time']
            interval_idx = state['interval_idx']
            
            if interval_idx == 0 and state['bkt_mastery_at_last_update'] < self.BKT_MASTERY_THRESHOLD:
                continue
            
            next_interval_days = self.INTERVALS[min(interval_idx, len(self.INTERVALS) - 1)]
            next_review_time = last_practice_time + timedelta(days=next_interval_days)
            
            if current_timestamp >= next_review_time:
                due_skills.append(skill_name)
        
        return due_skills