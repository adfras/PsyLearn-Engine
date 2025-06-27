# Psychology Tutor Engine

This project is an advanced intelligent tutoring system designed to personalize the learning experience. It combines two powerful AI paradigms: a predictive engine to model student knowledge and a generative engine to create dynamic learning content.

The core architecture is split into two main components:

1.  **The Proactive Tutor (Predictive)**: Uses student interaction logs and models like Bayesian Knowledge Tracing (BKT) and LightGBM to predict student performance, understand knowledge gaps, and determine when a student has mastered a skill.
2.  **The Reasoning Engine (Generative)**: Uses a fine-tuned T5 transformer model to generate high-quality, contextually relevant learning materials, such as plausible distractors for multiple-choice questions.

## Key Features

-   **Symbolic Student Modeling**: Leverages Bayesian Knowledge Tracing (BKT) to create an interpretable model of student skill mastery over time.
-   **Predictive Performance Modeling**: Uses a LightGBM classifier with rich features to accurately predict the probability of a student answering the next question correctly.
-   **Generative Content Creation**: A fine-tuned T5 model capable of generating plausible "distractor" answers for questions, enhancing the quality of assessments.
-   **Domain-Aware Data Pipeline**: A robust data processing pipeline that understands the difference between psychology-specific content and other knowledge domains, ensuring high-quality data for model training.

## The Data Pipeline

The project follows a multi-stage data pipeline to process raw information into model-ready training sets.

1.  **Data Ingestion & Normalization**: Raw data from various sources (Chain-of-Thought, Scholarly Q&A, Psychology Q&A) is cleaned, processed, and unified into a standard format (`normalized_questions.parquet`).
    -   *Script*: `src/reasoning_engine/prepare_data.py`

2.  **Embedding Generation**: The normalized questions are fed through a sentence-transformer model to create vector embeddings for each question, enabling semantic understanding.
    -   *Script*: `src/reasoning_engine/build_rag_index.py`

3.  **Domain-Aware Augmentation**: The embeddings are used to generate a high-quality training set for the generative model. This crucial step finds plausible distractors for questions, ensuring they are contextually and domain-relevent.
    -   *Process*: Run via the **Google Colab notebook** due to high computational requirements.
    -   *Output*: `distractor_generation_training_data_DOMAIN_AWARE_FIXED.parquet`

4.  **Model Training**:
    -   **Reasoning Engine (T5)**: The augmented distractor data is used to fine-tune a T5 model.
        -   *Script*: `src/reasoning_engine/finetune.py`
    -   **Proactive Tutor (LightGBM)**: Student interaction logs (from datasets like ASSISTments) are processed to create features, which are then used to train the performance prediction model.
        -   *Scripts*: `src/proactive_tutor/feature_engineering.py`, `notebooks/1_train_policy_model.ipynb`

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
    pip freeze > requirements.txt
    pip install -r requirements.txt
    ```

## How to Run the Pipeline

Execute the scripts in the following order to process the data and train the models.

1.  **Normalize the raw data:**
    ```bash
    python src/reasoning_engine/prepare_data.py
    ```

2.  **Generate vector embeddings for the questions:**
    ```bash
    python src/reasoning_engine/build_rag_index.py
    ```

3.  **Generate the domain-aware distractor training set:**
    -   Run the final, corrected notebook on **Google Colab**.
    -   Download the output (`..._FIXED.parquet`) and place it in the `data/4_training_sets/` directory.

4.  **Verify the quality of the generated training set:**
    ```bash
    pytest tests/test_augmented_data_quality.py -v
    ```

5.  **(Next Step) Fine-tune the T5 Reasoning Engine:**
    ```bash
    python src/reasoning_engine/finetune.py
    ```