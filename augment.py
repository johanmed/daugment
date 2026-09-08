"""
Main script of the package
To run: python augment.py --dataset-path <> --question-field <> --answer-field <> --local-dataset <> --output-path <>
Author: Johannes Medagbe
Copyright (c) 2026
"""

import argparse

import pandas as pd

from daugment.inference.embedding.extraction import extract
from daugment.inference.text.recommendation import recommend
from daugment.transforms.computation import propose_terms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", help="Path to dataset file or id")
    parser.add_argument("--output-path", help="Path to save augmented dataset at")
    parser.add_argument("--question-field", help="Question field in the dataset")
    parser.add_argument("--answer-field", help="Answer field in the dataset")
    parser.add_argument(
        "--local-dataset",
        default=False,
        help="Differentiate between local and HuggingFace dataset",
    )
    parser.add_argument(
        "--repo-word-model",
        default="adameubanks/YearlyWord2Vec",
        help="Path to repository offering word2vec model on HuggingFace Hub",
    )
    parser.add_argument(
        "--word-model-name",
        default="word2vec-2025/word2vec_2025.model",
        help="Specific version of word2vec model in the repository",
    )
    args = parser.parse_args()

    questions, unpacked_terms, answers, embeddings = extract(
        args.dataset_path,
        args.question_field,
        args.answer_field,
        args.repo_word_model,
        args.word_model_name,
        args.local_dataset,
    )
    new_terms = propose_terms(unpacked_terms, embeddings)
    new_dataset = recommend(questions, new_terms, answers)
    old_dataset = dict(zip(questions, answers))
    final_dataset = {**old_dataset, **new_dataset}
    df = pd.DataFrame(list(final_dataset.items()), columns=["question", "answer"])
    df.to_csv(args.output_path, header=True)
