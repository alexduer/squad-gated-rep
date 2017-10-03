import json
from typing import List, Dict, Tuple


class RawAnswer:
    def __init__(self, answer_start: int, text: str):
        self.answer_start = answer_start
        self.text = text

    @staticmethod
    def from_json(dict: Dict):
        return RawAnswer(dict['answer_start'], dict['text'])


class RawQuestion:
    def __init__(self, id: str, text: str, answers: List[RawAnswer]):
        self.id = id
        self.text = text
        self.answers = answers

    @staticmethod
    def from_json(dict: Dict):
        if 'answers' in dict:
            return RawQuestion(dict['id'], dict['question'],
                               [RawAnswer.from_json(answer) for answer in dict['answers']])
        else:
            return RawQuestion(dict['id'], dict['question'], [])


class RawParagraph:
    def __init__(self, context: str, questions: List[RawQuestion]):
        self.context = context
        self.questions = questions

    @staticmethod
    def from_json(dict: Dict):
        return RawParagraph(dict['context'], [RawQuestion.from_json(tasks) for tasks in dict['qas']])


class RawDocument:
    def __init__(self, title: str, paragraphs: List[RawParagraph]):
        self.title = title
        self.paragraphs = paragraphs

    @staticmethod
    def from_json(dict: Dict):
        return RawDocument(dict['title'], [RawParagraph.from_json(paragraph) for paragraph in dict['paragraphs']])


class RawDataset:
    def __init__(self, version: str, documents: List[RawDocument]):
        self.version = version
        self.documents = documents

    @staticmethod
    def from_json(dict: Dict):
        return RawDataset(dict['version'], [RawDocument.from_json(doc) for doc in dict['data']])

    def split_docs(self, split_ratio: float) -> Tuple['RawDataset', 'RawDataset']:
        assert 0.0 < split_ratio < 1.0
        n = int(split_ratio * 100)
        return (
            RawDataset(self.version, [doc for i, doc in enumerate(self.documents) if i % 100 >= n]),
            RawDataset(self.version, [doc for i, doc in enumerate(self.documents) if i % 100 < n])
        )


def load_raw_dataset(dataset_path: str) -> RawDataset:
    with open(dataset_path) as data_file:
        dataset_dict = json.load(data_file)

    return RawDataset.from_json(dataset_dict)
