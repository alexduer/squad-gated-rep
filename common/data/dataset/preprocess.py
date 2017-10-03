import os
from typing import List

from common.data.dataset.parser import StanfordService
from common.data.dataset.proc_dataset import Token, ProcAnswer, ProcDocument, ProcDataset, ProcQuestion, ProcParagraph
from common.data.dataset.raw_dataset import RawAnswer, RawParagraph, RawDataset, RawDocument
from common.data.evaluate import f1_score


def preprocess_answer(raw_answer: RawAnswer, raw_paragraph: RawParagraph,
                      answer_tokens: List[Token], par_tokens: List[Token]) -> ProcAnswer:
    ans_char_offset_start = raw_answer.answer_start
    ans_char_offset_end = raw_answer.answer_start + len(raw_answer.text)

    first_token_index = last_token_index = None
    for i in range(len(par_tokens)):
        token = par_tokens[i]
        token_char_offset_start = token.char_offset

        if token_char_offset_start <= ans_char_offset_start:
            first_token_index = i

        if token_char_offset_start < ans_char_offset_end:
            last_token_index = i

    if last_token_index is None or first_token_index is None:
        print('wtf')

    aligned_answer_tokens = par_tokens[first_token_index:(last_token_index + 1)]
    answer_text = raw_paragraph.context[aligned_answer_tokens[0].char_offset:aligned_answer_tokens[-1].char_offset_end()]

    return ProcAnswer(raw_answer, answer_tokens, aligned_answer_tokens, f1_score(answer_text, raw_answer.text))


def count_max_token_lengths(documents: List[ProcDocument]):
    max_par_tokens, max_qu_tokens, max_chars = 0, 0, 0

    for doc in documents:
        for par in doc.paragraphs:
            max_par_tokens = max([max_par_tokens, len(par.tokens)])
            max_chars = max([max_chars, max([len(tok.raw_text) for tok in par.tokens])])
            for qu in par.questions:
                max_qu_tokens = max([max_qu_tokens, len(qu.tokens)])
                max_chars = max([max_chars, max([len(tok.raw_text) for tok in qu.tokens])])

    return max_par_tokens, max_qu_tokens, max_chars


def preprocess(raw_dataset: RawDataset, data_dir: str) -> ProcDataset:
    stanford_service = StanfordService(os.path.join(data_dir, 'stanford-corenlp-full-2017-06-09/stanford-corenlp-full-2017-06-09'))
    documents = []

    for raw_doc in raw_dataset.documents:
        documents.append(preprocess_raw_doc(raw_doc, stanford_service))

    max_par_tokens, max_qu_tokens, max_chars = count_max_token_lengths(documents)
    return ProcDataset(raw_dataset, documents, max_par_tokens, max_qu_tokens, max_chars)


def preprocess_raw_doc(raw_doc: RawDocument, stanford_service: StanfordService) -> ProcDocument:
    doc_tokens = stanford_service.tokenize(raw_doc.title)
    paragraphs = []
    for raw_par in raw_doc.paragraphs:
        par_tokens = stanford_service.tokenize(raw_par.context)
        questions = []

        for raw_question in raw_par.questions:
            question_tokens = stanford_service.tokenize(raw_question.text)
            answers = []

            for raw_answer in raw_question.answers:
                answer_tokens = stanford_service.tokenize(raw_answer.text)
                answers.append(preprocess_answer(raw_answer, raw_par, answer_tokens, par_tokens))

            best_answer = max(answers, key=lambda a: a.token_level_f1)

            questions.append(ProcQuestion(raw_question, question_tokens, best_answer, answers))
        paragraphs.append(ProcParagraph(raw_par, par_tokens, questions))
    return ProcDocument(raw_doc, doc_tokens, paragraphs)
