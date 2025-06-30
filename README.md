# Psychology Tutor Engine

This project is an advanced intelligent tutoring system designed to personalize the learning experience. It combines two powerful AI paradigms: a predictive engine to model student knowledge and a generative engine to create dynamic learning content.

The core architecture is split into two main components:

1.  **The Proactive Tutor (Predictive)**: Uses student interaction logs and models like Bayesian Knowledge Tracing (BKT) and LightGBM to predict student performance, understand knowledge gaps, and determine when a student has mastered a skill.
2.  **The Reasoning Engine (Generative)**: Uses a fine-tuned T5 transformer model to generate high-quality, contextually relevant learning materials, such as plausible distractors for multiple-choice questions.

## Project Status (As of Latest Commit)

*   ✅ **Reasoning Engine (T5 Distractor Generator):** Complete. A production-ready model has been fine-tuned, validated, and is available in the `models/` directory.
*   ⏳ **Proactive Tutor (LGBM Performance Predictor):** Pending. The next phase of the project will focus on building this component.

## Key Features

-   **Symbolic Student Modeling**: Leverages Bayesian Knowledge Tracing (BKT) to create an interpretable model of student skill mastery over time.
-   **Predictive Performance Modeling**: Uses a LightGBM classifier with rich features to accurately predict the probability of a student answering the next question correctly.
-   **Generative Content Creation**: A fine-tuned T5 model capable of generating plausible "distractor" answers for questions, enhancing the quality of assessments.
-   **Domain-Aware Data Pipeline**: A robust data processing pipeline that understands the difference between psychology-specific content and other knowledge domains, ensuring high-quality data for model training.

## The Data & Model Pipeline

The project follows a multi-stage pipeline to process raw information into trained, functional models.

1.  **Data Ingestion & Normalization**: Raw data from various sources is cleaned, processed, and unified into a standard format (`normalized_questions.parquet`).
    -   *Script*: `src/reasoning_engine/prepare_data.py`

2.  **Embedding Generation**: The normalized questions are fed through a sentence-transformer model to create vector embeddings for each question, enabling semantic understanding.
    -   *Script*: `src/reasoning_engine/build_rag_index.py` (Run on Colab for GPU acceleration).

3.  **Domain-Aware Augmentation**: The embeddings are used to generate a high-quality training set for the generative model. This crucial step finds plausible distractors for questions, ensuring they are contextually and domain-relevant.
    -   *Process*: Run via the **`notebooks/Distract_Generator.ipynb`** notebook on Google Colab due to high computational requirements.
    -   *Output*: `distractor_generation_training_data_DOMAIN_AWARE_FIXED.parquet`

4.  **Fine-Tuning the Reasoning Engine (T5 Model)**: The augmented distractor data is used to fine-tune a T5 model. The training script cleans the data on-the-fly and uses special tokens for robust learning.
    -   *Process*: Run via a **Google Colab notebook** (based on the `src/reasoning_engine/finetune.py` logic) to leverage GPUs.
    -   *Output*: The final `distractor_generator_t5_production` model.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd psychology-tutor-engine
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    *(Note: You should create a `requirements.txt` file for your project)*
    ```bash
    pip install torch transformers sentencepiece pandas
    ```

## How to Run the Pipeline & Use the Model

The model training is compute-intensive and best performed on Google Colab. The final model is included in the repository.

1.  **(To Retrain)** Follow the pipeline steps 1-4, using Google Colab for the GPU-dependent stages.
2.  **(To Use the Pre-trained Model)** The final model is located at `models/distractor_generator_t5_production/`. You can test it locally using the `tests/test_local_model.py` script.
    ```bash
    # Make sure you are in the project root with your venv active
    python tests/test_local_model.py
    ```