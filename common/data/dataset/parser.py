import os
from time import sleep
from typing import List

from corenlp import CoreNLPClient

from common.data.dataset.proc_dataset import Token


class StanfordToken:
    def __init__(self, sentence_index: int, token_index: int, characterOffsetBegin: int, characterOffsetEnd: int,
                 word: str, originalText: str):
        self.sentence_index = sentence_index
        self.token_index = token_index
        self.characterOffsetBegin = characterOffsetBegin
        self.characterOffsetEnd = characterOffsetEnd
        self.word = word
        self.originalText = originalText

    @staticmethod
    def from_proto(token, sentence_index: int):
        return StanfordToken(
            sentence_index,
            token.beginIndex,
            token.beginChar,
            token.endChar,
            token.word,
            token.originalText
        )


class StanfordSentence:
    def __init__(self, index: int, tokens: List[StanfordToken]):
        self.index = index
        self.tokens = tokens

    @staticmethod
    def from_proto(sentence):
        return StanfordSentence(sentence.sentenceIndex, [StanfordToken.from_proto(t, sentence.sentenceIndex)
                                                         for t in sentence.token])


class StanfordDocument:
    def __init__(self, sentences: List[StanfordSentence]):
        self.sentences = sentences

    @staticmethod
    def from_proto(annotated_result):
        return StanfordDocument([StanfordSentence.from_proto(s) for s in annotated_result.sentence])


class StanfordCorefMention:
    def __init__(self, sentence_index, begin_word_index: int, end_word_index: int):
        self.sentence_index = sentence_index
        self.begin_word_index = begin_word_index
        self.end_word_index = end_word_index

    @staticmethod
    def from_proto(mention):
        return StanfordCorefMention(mention.sentenceIndex, mention.beginIndex, mention.endIndex)


class StanfordCorefChain:
    def __init__(self, rep_mention: StanfordCorefMention, all_mentions: List[StanfordCorefMention]):
        self.rep_mention = rep_mention
        self.ordered_mentions = sorted(all_mentions, key=lambda m: m.sentence_index*1e6 + m.begin_word_index)

    @staticmethod
    def from_proto(coref_chain):
        all_mentions = [StanfordCorefMention.from_proto(mention) for mention in coref_chain.mention]
        return StanfordCorefChain(all_mentions[coref_chain.representative], all_mentions)


class StanfordService:
    def __init__(self, parser_path: str):
        os.environ['JAVANLP_HOME'] = parser_path
        print('starting CoreNLP server with JAVANLP_HOME {}'.format(parser_path))
        self.nlp = CoreNLPClient(annotators="tokenize ssplit".split(), timeout=1000000)

    def tokenize(self, text: str) -> List[Token]:
        for _ in range(10):
            try:
                annotated_result = self.nlp.annotate(text)
                stanford_document = StanfordDocument.from_proto(annotated_result)
                return StanfordService.idiomatic_tokens(stanford_document)
            except:
                print('exception while annotating result')
                sleep(10)

    @staticmethod
    def idiomatic_tokens(doc: StanfordDocument):
        stanford_tokens = [token for sentence in doc.sentences for token in sentence.tokens]
        return [StanfordService.idiomatic_token(token, index) for index, token in enumerate(stanford_tokens)]

    @staticmethod
    def idiomatic_token(token: StanfordToken, token_index: int) -> Token:
        return Token(token.originalText, token_index, token.characterOffsetBegin)
