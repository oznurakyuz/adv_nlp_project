import itertools
import os
import re
from typing import List, Tuple, Any

import conllu
import zeyrek
from tqdm import tqdm

from string_utils import remove_punctuation, remove_repeating_spaces, filter_allowed_characters, to_lower

MORPH_ANALYZER = zeyrek.MorphAnalyzer()
DATA_PATH = '../data_resource'


def converter(analysis: List) -> tuple[Any, tuple[Any, ...]]:
    res = set()
    for morpheme in analysis:
        big_regex = re.compile('|'.join(map(re.escape, morpheme.morphemes)))
        morpheme = morpheme.formatted.split(" ")[-1]  # ['[gelmek:Verb]', 'gel:Verb|ecek:FutPart→Adj']
        morpheme = big_regex.sub("", morpheme)
        morpheme = remove_punctuation(morpheme)
        morpheme = remove_repeating_spaces(morpheme)
        morpheme = morpheme.replace(" ", "-")
        if morpheme != '':
            res.add(morpheme)

    return (analysis[0].word, tuple(res)) if len(res) else None


def morphomize(str_: str) -> Tuple:
    str_ = to_lower(str_)
    str_ = filter_allowed_characters(str_)
    analysis = MORPH_ANALYZER.analyze(str_)
    analysis = tuple(filter(None, map(converter, filter(len, analysis))))
    return analysis


def save_raw_data(corpus_chunk, file_name="raw_corpus"):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    with open(f"{DATA_PATH}/{file_name}.txt", "a+", encoding="utf-8") as file:
        file.write("\n".join([f"{str_[0]}:{'+'.join(str_[1])}" for str_ in corpus_chunk]))
        file.write("\n")


def save_structured_data(max_segmentation_length, chunk_size, read_file="raw_corpus", write_file="structured_corpus"):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    processed = list()
    with open(f"{DATA_PATH}/{read_file}.txt", "r", encoding="utf-8") as rfile:
        for line in rfile:
            line = line.strip()
            word, morphemes = line.split(":")
            morphemes = morphemes.split("+")
            morphemes.extend(['###'] * (max_segmentation_length - len(morphemes) - 1))
            morphemes.append("-".join(list(word)))
            processed.append(f"{word}:{'+'.join(morphemes)}")

            if len(processed) % chunk_size == 0:
                with open(f"{DATA_PATH}/{write_file}_{max_segmentation_length}segment.txt", "a+",
                          encoding="utf-8") as wfile:
                    wfile.write("\n".join(processed))
                    wfile.write("\n")
                    processed = list()
        else:
            with open(f"{DATA_PATH}/{write_file}_{max_segmentation_length}segment.txt", "a+",
                      encoding="utf-8") as wfile:
                wfile.write("\n".join(processed))
                wfile.write("\n")


if __name__ == "__main__":
    # todo initialize corpus when datasets are obtained.
    vocab = set()
    data_files = ["turkish-ner-train.conllu", "turkish-ner-dev.conllu", "turkish-ner-test.conllu"]
    for data_file in data_files:
        data_file = open(f"{DATA_PATH}/datasets/{data_file}", "r", encoding="utf-8")
        for tokenlist in tqdm(conllu.parse_incr(data_file)):
            vocab.update([str(token) for token in tokenlist])

    vocab = list(vocab)
    chunk_size = 1000
    max_segmentation_length = 10
    for i in tqdm(range(0, len(vocab), chunk_size)):
        corpus_chunk = set(itertools.chain(*map(morphomize, vocab[i:i + chunk_size])))
        if len(corpus_chunk):
            segmentation_length = len(max(corpus_chunk, key=lambda l: len(l[1]))[1])
            if segmentation_length > max_segmentation_length:
                max_segmentation_length = segmentation_length
            print(f"processed {i + chunk_size}/{len(vocab)} max_segmentation_length: {max_segmentation_length}")
            save_raw_data(corpus_chunk, file_name="raw_news")

    save_structured_data(max_segmentation_length, chunk_size, read_file="raw_news", write_file="structured_news")
