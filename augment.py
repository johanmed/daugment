"""
Main script of the package
To run: python augment.py
Author: Johannes Medagbe
Copyright (c) 2026
"""

import argparse
import os

import dspy
import pandas as pd
from dotenv import load_dotenv

from daugment.inference.embedding.extraction import extract
from daugment.inference.text.recommendation import recommend
from daugment.transforms.computation import propose_terms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-file", default=".env", help="Path to file with environment file"
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=args.env_file)
    dataset_path = os.environ["DATASET_PATH"]
    question_field = os.environ["QUESTION_FIELD"]
    answer_field = os.environ["ANSWER_FIELD"]
    local_dataset = bool(int(os.environ["LOCAL_DATASET"]))
    output_path = os.environ["OUTPUT_PATH"]
    repo_word_model = os.environ["REPO_WORD_MODEL"]
    word_model_name = os.environ["WORD_MODEL_NAME"]
    llm_name = os.environ["LLM_NAME"]
    api_key = os.environ["API_KEY"]

    llm = dspy.LM(
        llm_name,
        api_key=api_key,
        max_tokens=5_000,
        temperature=1,
        cache=False,
        verbose=False,
    )
    dspy.configure(lm=llm)

    questions, unpacked_terms, answers, embeddings = extract(
        dataset_path,
        question_field,
        answer_field,
        repo_word_model,
        word_model_name,
        local_dataset,
    )
    new_terms = propose_terms(unpacked_terms, embeddings)
    new_dataset = recommend(questions, new_terms, answers)
    old_dataset = dict(zip(questions, answers))
    final_dataset = {**old_dataset, **new_dataset}
    df = pd.DataFrame(list(final_dataset.items()), columns=["question", "answer"])
    df.to_csv(output_path, header=True)
