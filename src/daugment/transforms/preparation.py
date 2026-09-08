"""Module with constructs to prepare data for augmentation"""

import pandas as pd
from datasets import load_dataset


def read_dataset(
    dataset_path: str,
    question_field: str,
    answer_field: str,
    local_path: bool = False,
) -> dict[str, str]:
    if local_path is False:
        dataset = load_dataset(dataset_path)
        train_set = dataset["train"]
        questions = list(train_set[question_field])
        answers = list(train_set[answer_field])
    else:
        df = pd.read_csv(dataset_path)
        questions = df[question_field]
        answers = df[answer_field]
    return dict(zip(questions, answers))
