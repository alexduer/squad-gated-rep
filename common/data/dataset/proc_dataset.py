from typing import List

from common.data.dataset.raw_dataset import RawAnswer, RawQuestion, RawParagraph, RawDocument, RawDataset


class Token:
    def __init__(self, raw_text: str, word_offset: int, char_offset: int):
        self.raw_text = raw_text
        self.word_offset = word_offset
        self.char_offset = char_offset

    def char_offset_end(self):
        return self.char_offset + len(self.raw_text)


class ProcAnswer:
    def __init__(self, raw: RawAnswer, original_tokens: List[Token], aligned_tokens: List[Token],
                 token_level_f1: float):
        self.raw = raw
        self.original_tokens = original_tokens
        self.aligned_tokens = aligned_tokens
        self.token_level_f1 = token_level_f1


class ProcQuestion:
    def __init__(self, raw: RawQuestion, tokens: List[Token], answer: ProcAnswer, all_answers: List[ProcAnswer]):
        self.raw = raw
        self.tokens = tokens
        self.answer = answer
        self.all_answers = all_answers


class ProcParagraph:
    def __init__(self, raw: RawParagraph, tokens: List[Token], questions: List[ProcQuestion]):
        self.raw = raw
        self.tokens = tokens
        self.questions = questions


class ProcDocument:
    def __init__(self, raw: RawDocument, tokens: List[Token], paragraphs: List[ProcParagraph]):
        self.raw = raw
        self.tokens = tokens
        self.paragraphs = paragraphs


class ProcDataset:
    def __init__(self, raw: RawDataset, documents: List[ProcDocument],
                 max_par_tokens: int, max_qu_tokens: int, max_chars: int):
        self.raw = raw
        self.documents = documents
        self.max_par_tokens = max_par_tokens
        self.max_qu_tokens = max_qu_tokens
        self.max_chars = max_chars
