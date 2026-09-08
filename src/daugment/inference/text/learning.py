"""Module with constructs to infer links between questions and answers using state-of-art LLM"""

import dspy


class Infer(dspy.Signature):
    """
    Analyze questions and corresponding answers.
    Identify terms in the questions that hold more semantic weight and directly associate with the corresponding answer across pairs of questions and answers.
    Avail the list of key terms in each question that you linked to the corresponding answer after thorough semantic and pattern analysis.
    """

    questions: list[str] = dspy.InputField("List of questions in the dataset")
    answer: list[str] = dspy.InputField("List of corresponding responses")
    key_terms: list[list[str]] = dspy.OutputField(
        "List of key terms in each question that were most influential and motivated the true answer inferred for the question"
    )


def learn(
    dataset: dict[str, str], batch_size: int = 100
) -> tuple[dict[tuple[str], str], list[str]]:
    """Learn key terms in questions based on answers"""
    questions = list(dataset.keys())
    infer = dspy.Predict(Infer)
    relationships = {}
    for ind in range(len(dataset), batch_size):
        batch_questions = questions[ind : ind + batch_size + 1]
        batch_answers = [dataset[question] for question in batch_questions]
        key_terms = infer(questions=batch_questions, answers=batch_answers).get(
            "key_terms"
        )
        for key_term, answer in zip(key_terms, batch_answers):
            relationships[tuple(key_term)] = answer
    return relationships, questions
