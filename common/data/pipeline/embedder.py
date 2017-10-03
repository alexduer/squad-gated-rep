from typing import List, Callable

import numpy as np
import os

from common.data.dataset.proc_dataset import Token, ProcDataset

UNKNOWN_KEY = 'UNK'
EMBEDDING_SIZE = 300


class EmbeddingService:
    def __init__(self, tokens: List[str], embedding_matrix: np.ndarray):
        self.tokens = [UNKNOWN_KEY]
        self.tokens.extend(tokens)

        self.token2index = {token: index for index, token in enumerate(self.tokens)}
        self.embedding_matrix = np.insert(embedding_matrix, 0, np.zeros((1,)), axis=0)
        self.embedding_dim = embedding_matrix.shape[1]

    def indices(self, tokens: List[Token]) -> List[int]:
        return [self.str_index(token.raw_text) for token in tokens]

    def str_indices(self, token_texts: List[str]) -> List[int]:
        return [self.str_index(text) for text in token_texts]

    def str_index(self, token_text: str) -> int:
        if token_text in self.token2index:
            return self.token2index[token_text]
        else:
            return self.token2index[UNKNOWN_KEY]


def load_word_embedder(datasets: List[ProcDataset], data_dir: str) -> EmbeddingService:
    token_texts = set()

    for dataset in datasets:
        for doc in dataset.documents:
            token_texts.update([token.raw_text for token in doc.tokens])
            token_texts.update([token.raw_text.lower() for token in doc.tokens])

            for par in doc.paragraphs:
                token_texts.update([token.raw_text for token in par.tokens])
                token_texts.update([token.raw_text.lower() for token in par.tokens])

                for qu in par.questions:
                    token_texts.update([token.raw_text for token in qu.tokens])
                    token_texts.update([token.raw_text.lower() for token in qu.tokens])

                    token_texts.update([token.raw_text for token in qu.answer.aligned_tokens])
                    token_texts.update([token.raw_text.lower() for token in qu.answer.aligned_tokens])
                    token_texts.update([token.raw_text for token in qu.answer.original_tokens])
                    token_texts.update([token.raw_text.lower() for token in qu.answer.original_tokens])

    return load_embedder(data_dir, lambda text: text in token_texts)


def load_char_embedder(data_dir: str) -> EmbeddingService:
    return load_embedder(data_dir, lambda text: len(text) == 1)


def load_embedder(data_dir: str, criterion: Callable[[str], bool]) -> EmbeddingService:
    word_vector_file = os.path.join(data_dir, 'word-vectors/glove.840B.300d.txt')
    embedding_list = []

    with open(word_vector_file, encoding='UTF-8') as file:
        for line in file:
            elements = line.split(' ')

            if len(elements) == EMBEDDING_SIZE + 1:
                token_text = elements[0]

                if criterion(token_text):
                    embedding = [float(element) for element in elements[1:]]
                    embedding_list.append((token_text, np.asarray(embedding)))

    print('loaded ', len(embedding_list), ' embeddings from file.')

    embedding_tokens = []
    embedding_matrix = np.zeros((len(embedding_list), EMBEDDING_SIZE))
    for index, (word, vec) in enumerate(embedding_list):
        embedding_tokens.append(word)
        embedding_matrix[index] = vec

    return EmbeddingService(embedding_tokens, embedding_matrix)


def load_small_char_embedder(data_dir: str, char_embedding_size: int) -> EmbeddingService:
    chars = set()

    with open(os.path.join(data_dir, 'train-v1.1.json'), 'r') as file:
        chars.update(file.read().replace('\n', ''))

    with open(os.path.join(data_dir, 'dev-v1.1.json'), 'r') as file:
        chars.update(file.read().replace('\n', ''))

    return EmbeddingService(sorted([c for c in chars]), np.random.normal(0.0, 0.01, (len(chars), char_embedding_size)))