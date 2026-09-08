"""Module with constructs to infer links between questions and answers using state-of-art LLM"""

import dspy


class InferTerms(dspy.Signature):
    """
    Analyze questions and corresponding answers.
    Identify terms in the questions that hold more semantic weight and directly associate with the corresponding answer across pairs of questions and answers.
    Avail the list of key terms in each question that you linked to the corresponding answer after thorough semantic and pattern analysis.
    """

    questions: list[str] = dspy.InputField(desc="List of questions in the dataset")
    answers: list[str] = dspy.InputField(desc="List of corresponding responses")
    key_terms: list[list[str]] = dspy.OutputField(
        desc="List of key terms in each question that were most influential and motivated the true answer inferred for the question"
    )


class CategorizeQuestions(dspy.Signature):
    """
    Analyze questions and group them into specific categories based on meaning of the task at hand.
    Return dictionary where categories are keys and list of questions falling under the specific categories are values.
    Similar questions should never be assigned to different categories.
    """

    questions: list[str] = dspy.InputField(
        desc="List of questions to study and infer categories from"
    )
    categories: dict[str, list[str]] = dspy.OutputField(
        desc="Inferred grouping of questions based on semantic similarity of the task at hand"
    )


def learn_terms(
    dataset: dict[str, str], batch_size: int = 100
) -> tuple[dict[tuple[str], str], list[str]]:
    """Learn key terms in questions based on answers"""
    questions = list(dataset.keys())
    infer_terms = dspy.Predict(InferTerms)
    relationships = {}
    for ind in range(0, len(dataset), batch_size):
        batch_questions = questions[ind : ind + batch_size + 1]
        batch_answers = [dataset[question] for question in batch_questions]
        key_terms = infer_terms(questions=batch_questions, answers=batch_answers).get(
            "key_terms"
        )
        for key_term, answer in zip(key_terms, batch_answers):
            relationships[tuple(key_term)] = answer
    return relationships, questions


def categorize_dataset(
    dataset: dict[str, str], num_subdatasets: int = 10, batch_size: int = 100
) -> list[list[tuple[str]]]:
    """Categorize questions and answers based on semantic similarity"""
    questions = list(dataset.keys())
    categorize_questions = dspy.Predict(CategorizeQuestions)
    categorized_dataset = {}
    for ind in range(0, len(dataset), batch_size):
        batch_questions = questions[ind : ind + batch_size + 1]
        categories = categorize_questions(questions=batch_questions).get("categories")
        for category in categories:
            if category not in categorized_dataset:
                subquestions = categories[category]
                subanswers = [dataset[question] for question in subquestions]
                categorized_dataset[category] = dict(zip(subquestions, subanswers))
            else:
                subquestions = categories[category]
                subanswers = [dataset[question] for question in subquestions]
                categorized_dataset[category] = {
                    **categorized_dataset[category],
                    **dict(zip(subquestions, subanswers)),
                }
    new_datasets = [[] for _ in num_subdatasets]
    for category in categorized_dataset.values():
        for pair in category.items():
            n_elements = len(pair) // num_subdatasets
            for ind1, ind2 in enumerate(range(0, len(pairs) + 1, n_elements)):
                new_datasets[ind1].append(pair[ind2 : ind2 + n_elements + 1])
    return new_datasets
