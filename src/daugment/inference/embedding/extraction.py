"""Module with constructs to generate word embeddings for key terms using word2vec from HuggingFace Hub"""

import numpy as np
from daugment.inference.text.learning import learn
from daugment.transforms.preparation import read_dataset
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download


def multi_embed(
    terms: list[str], repository_id: str, model_name: str
) -> tuple[list[str], np.ndarray]:
    """Embed a list of terms using a word2vec model"""
    model_path = hf_hub_download(repo_id=repository_id, filename=model_name)
    model = KeyedVectors.load(model_path)
    vectors = model.wv
    final_terms = []
    embeddings = []
    for term in terms:
        if term in vectors:
            embeddings.append(vectors[term])
            final_terms.append(term)
    return final_terms, np.vstack(embeddings)


def extract(
    dataset_path: str,
    question_field: str,
    answer_field: str,
    repository_id: str,
    model_name: str,
    local_path: bool = False,
) -> tuple[list[str], list[str], list[str], np.ndarray]:
    """Extract from a dataset most influential terms in questions, their embeddings and corresponding answers"""
    dataset = read_dataset(dataset_path, question_field, answer_field, local_path)
    terms_answers, questions = learn(dataset)
    terms = list(terms_answers.keys())
    unpacked_terms = [term for sublist in terms for term in sublist]
    answers = list(terms_answers.values())
    final_terms, embeddings = multi_embed(unpacked_terms, repository_id, model_name)
    return questions, final_terms, answers, embeddings
