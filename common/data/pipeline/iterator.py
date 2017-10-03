from collections import Iterator
from random import shuffle
from typing import List, Tuple

import numpy as np

from common.data.dataset.proc_dataset import ProcDataset, ProcDocument, ProcParagraph, ProcQuestion, Token
from common.data.pipeline.embedder import EmbeddingService


class ProcDatasetIter(Iterator):
    def __init__(self, dataset: ProcDataset, shuffle_samples: bool = True, bucket_size: int = 4096):
        self.dataset = dataset
        self.shuffle_samples = shuffle_samples
        self.bucket_size = bucket_size

        self.curr_order = []
        self.curr_index = 0
        self.sorted_samples = self.generate_sorted_samples()
        self.reset_iter()

    def generate_sorted_samples(self) -> List[Tuple[ProcDocument, ProcParagraph, ProcQuestion]]:
        samples = []
        for doc in self.dataset.documents:
            for par in doc.paragraphs:
                for qu in par.questions:
                    samples.append((doc, par, qu))
        samples = sorted(samples, key=lambda s: len(s[1].tokens))
        return samples

    def reset_iter(self):
        num_samples = len(self.sorted_samples)
        order = [i for i in range(num_samples)]
        shuffle(order)

        num_buckets = (num_samples + self.bucket_size - 1) // self.bucket_size
        for i in range(num_buckets):
            bucket_start, bucket_end = i*self.bucket_size, min((i+1)*self.bucket_size, num_samples)
            order[bucket_start:bucket_end] = sorted(order[bucket_start:bucket_end],
                                                    key=lambda i: len(self.sorted_samples[i][1].tokens))

        self.curr_order = order
        self.curr_index = 0

    def __next__(self) -> Tuple[ProcDocument, ProcParagraph, ProcQuestion]:
        if self.curr_index >= len(self.curr_order):
            print('auto-resetting iter')
            self.reset_iter()

        self.curr_index += 1
        return self.sorted_samples[self.curr_order[self.curr_index - 1]]

    def num_valid_samples(self, max_par_words: int, filter_short_questions: bool) -> int:
        return len([qu
                    for doc, par, qu in self.sorted_samples
                    if qu.answer.aligned_tokens[-1].word_offset < max_par_words and
                    (len(qu.raw.text) >= 4 or not filter_short_questions)])


class SampleBatch:
    def __init__(self, qu_ids: List[str],
                 par_words: np.ndarray, par_num_words: np.ndarray,
                 par_chars: np.ndarray, par_num_chars: np.ndarray,
                 qu_words: np.ndarray, qu_num_words: np.ndarray,
                 qu_chars: np.ndarray, qu_num_chars: np.ndarray,
                 answer_labels: np.ndarray):
        self.qu_ids = qu_ids
        self.par_words = par_words
        self.par_num_words = par_num_words
        self.par_chars = par_chars
        self.par_num_chars = par_num_chars
        self.qu_words = qu_words
        self.qu_num_words = qu_num_words
        self.qu_chars = qu_chars
        self.qu_num_chars = qu_num_chars
        self.answer_labels = answer_labels


class BatchIter(Iterator):
    def __init__(self, dataset: ProcDataset, word_embedder: EmbeddingService, char_embedder: EmbeddingService,
                 batch_size: int, max_par_length: int, max_qu_length: int, max_char_length: int,
                 discard_invalid_samples: bool, filter_short_questions: bool, iter_bucket_size: int):
        self.iterator = ProcDatasetIter(dataset, bucket_size=iter_bucket_size)
        self.word_embedder = word_embedder
        self.char_embedder = char_embedder
        self.batch_size = batch_size
        self.max_par_length = max_par_length
        self.max_qu_length = max_qu_length
        self.max_char_length = max_char_length
        self.discard_invalid_samples = discard_invalid_samples
        self.filter_short_questions = filter_short_questions

    def num_samples(self):
        return len(self.iterator.sorted_samples)

    def num_valid_samples(self):
        return self.iterator.num_valid_samples(self.max_par_length, self.filter_short_questions)

    def num_batches(self) -> int:
        return (self.num_samples() + self.batch_size - 1) // self.batch_size

    def num_valid_batches(self) -> int:
        return (self.num_valid_samples() + self.batch_size - 1) // self.batch_size

    def __next__(self) -> SampleBatch:
        batch_samples = []
        for _ in range(self.batch_size):
            doc, par, qu = self.iterator.__next__()

            while (self.discard_invalid_samples and qu.answer.aligned_tokens[-1].word_offset >= self.max_par_length) \
                    or (self.filter_short_questions and len(qu.raw.text) < 4):
                doc, par, qu = self.iterator.__next__()

            batch_samples.append((doc, par, qu))

        batch_par_len = min([self.max_par_length, max(map(lambda x: len(x[1].tokens), batch_samples))])
        batch_qu_len = min([self.max_qu_length, max(map(lambda x: len(x[2].tokens), batch_samples))])
        batch_char_len = min([self.max_char_length, max([len(t.raw_text)
                                                         for doc, par, qu in batch_samples
                                                         for t in [*par.tokens, *qu.tokens]])])

        sample = SampleBatch(
            qu_ids=self.batch_size * [''],
            par_words=np.zeros((self.batch_size, batch_par_len), dtype=np.int32),
            par_num_words=np.zeros((self.batch_size), dtype=np.int32),
            par_chars=np.zeros((self.batch_size, batch_par_len, batch_char_len), dtype=np.int32),
            par_num_chars=np.zeros((self.batch_size, batch_par_len), dtype=np.int32),
            qu_words=np.zeros((self.batch_size, batch_qu_len), dtype=np.int32),
            qu_num_words=np.zeros((self.batch_size), dtype=np.int32),
            qu_chars=np.zeros((self.batch_size, batch_qu_len, batch_char_len), dtype=np.int32),
            qu_num_chars=np.zeros((self.batch_size, batch_qu_len), dtype=np.int32),
            answer_labels=np.zeros((self.batch_size, 2), dtype=np.int32),
        )

        for j, (doc, par, qu) in enumerate(batch_samples):
            sample.qu_ids[j] = qu.raw.id

            par_num_words = min(len(par.tokens), batch_par_len)
            sample.par_words[j, 0:par_num_words] = np.asarray(self.word_embedder.indices(par.tokens))[0:par_num_words]
            sample.par_num_words[j] = par_num_words
            sample.par_chars[j, 0:par_num_words, 0:batch_char_len] = \
                self.char_lookup_matrix(par.tokens, par_num_words, batch_char_len)
            sample.par_num_chars[j, 0:par_num_words] = np.asarray([min(len(tok.raw_text), batch_char_len)
                                                                   for tok in par.tokens])[0:par_num_words]

            qu_num_words = min(len(qu.tokens), batch_qu_len)
            sample.qu_words[j, 0:qu_num_words] = np.asarray(self.word_embedder.indices(qu.tokens))[0:qu_num_words]
            sample.qu_num_words[j] = qu_num_words
            sample.qu_chars[j, 0:qu_num_words, 0:batch_char_len] = \
                self.char_lookup_matrix(qu.tokens, qu_num_words, batch_char_len)
            sample.qu_num_chars[j, 0:qu_num_words] = np.asarray([min(len(tok.raw_text), batch_char_len)
                                                                 for tok in qu.tokens])[0:qu_num_words]

            first_answer_word = min([batch_par_len, qu.answer.aligned_tokens[0].word_offset])
            second_answer_word = min([batch_par_len, qu.answer.aligned_tokens[-1].word_offset])
            sample.answer_labels[j] = np.array([first_answer_word, second_answer_word])

        return sample

    def char_lookup_matrix(self, tokens: List[Token], num_words: int, batch_char_len: int):
        matrix = np.zeros((num_words, batch_char_len))
        num_tokens = min(len(tokens), num_words)

        for i in range(num_tokens):
            curr_word = tokens[i].raw_text
            num_chars = min([batch_char_len, len(curr_word)])
            matrix[i, 0:num_chars] = np.asarray(self.char_embedder.str_indices(curr_word))[0:num_chars]

        return matrix
