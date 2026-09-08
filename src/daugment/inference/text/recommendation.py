"""Module with constructs to recommend new data from proposed terms"""

import dspy


class Contextualize(dspy.Signature):
    """
    Analyze the list of questions and infer general informations about the task they aimed to address.
    The goal is to extract the context of a task from a limited list of questions to generate more questions that are relevant for the task. So be very thorough and detailed in your analysis and inference.
    """

    questions: list[str] = dspy.InputField(
        desc="List of questions to infer general informations about the task at hand from"
    )
    context: str = dspy.OutputField(
        desc="Context of most questions in the task: domain, objective, type of data, etc."
    )


class Produce(dspy.Signature):
    """
    Generate 10 new questions of highly good quality from the selection of terms that fit the context of the task at hand and are different from the previous questions.
    """

    terms: list[str] = dspy.InputField(
        desc="List of terms to explore combinations from"
    )
    context: str = dspy.InputField(
        desc="Task context to ground propositions of questions in"
    )
    previous_questions: list = dspy.InputField(
        desc="List of questions already available to not repeat"
    )
    new_questions: list[str] = dspy.OutputField(
        desc="List of new questions built from term combinations that make sense and are relevant for the task context"
    )


class Derive(dspy.Signature):
    """
    Produce answers to the questions using general and accumulated knowledge over multiple rounds of sustained inference.
    """

    questions: list[str] = dspy.InputField(desc="List of questions to address")
    priors: dict[str, str] = dspy.InputField(
        desc="Knowledge accumulated through previous rounds of question-answering. Keys are questions and values are answers."
    )
    answers: list[str] = dspy.OutputField(
        desc="List of corresponding answers based on general knowledge and priors"
    )


def recommend(
    questions: list[str], terms: list[str], answers: list[str], n_reps: int = 10
) -> dict[str, str]:
    """
    Recommend new pairs of questions and answers based on selection of terms and context
    Size of new dataset depends on n_reps: typically n_reps * 10
    """
    contextualize = dspy.Predict(Contextualize)
    context = contextualize(questions=questions).get("context")
    produce = dspy.Predict(Produce)
    derive = dspy.Predict(Derive)
    priors = dict(zip(questions, answers))
    question_propositions = []
    answer_propositions = []
    for rep in range(n_reps):
        new_questions = produce(
            terms=terms, context=context, previous_questions=question_propositions
        ).get("new_questions")
        new_answers = derive(questions=new_questions, priors=priors).get("answers")
        question_propositions.extend(new_questions)
        answer_propositions.extend(new_answers)
    return dict(zip(question_propositions, answer_propositions))
