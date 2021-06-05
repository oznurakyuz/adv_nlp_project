import os
import re
from typing import List, Tuple, Any

import zeyrek

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
    analysis = MORPH_ANALYZER.analyze(str_)  # {analysis[0].word}
    analysis = tuple(filter(None, map(converter, filter(len, analysis))))
    return analysis


def save_raw_data(corpus_chunk, file_name="raw_corpus"):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    with open(f"{DATA_PATH}/{file_name}.txt", "a+", encoding="utf-8") as file:
        file.write("\n".join([f"{str_[0]}:{'+'.join(str_[1])}" for str_ in corpus_chunk]))
        file.write("\n")


def save_structured_data(max_segmentation_length, read_file="raw_corpus", write_file="structured_corpus"):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    with open(f"{DATA_PATH}/{read_file}.txt", "r", encoding="utf-8") as rfile:
        processed = list()
        for line in rfile:
            line = line.strip()
            word, morphemes = line.split(":")
            morphemes = morphemes.split("+")
            morphemes.extend(['###'] * (max_segmentation_length - len(morphemes) - 1))
            morphemes.append("-".join(list(word)))
            processed.append(f"{word}:{'+'.join(morphemes)}")

            if len(processed) % 5 == 0:
                with open(f"{DATA_PATH}/{write_file}_{max_segmentation_length}segment.txt", "a+",
                          encoding="utf-8") as wfile:
                    wfile.write("\n".join(processed))
                    wfile.write("\n")


if __name__ == "__main__":
    # todo initialize corpus when datasets are obtained.
    corpus = [
        'benim geleceğim',
        'iyi bir fikir olmayabilir',
        'en çok okunan',
        'Şemdinli de çatışma: 5 şehit Hakkari nin Şemdinli ilçesine bağlı ortaklar köyü kırsalında çıkan çatışmada 5 asker şehit oldu 1 asker yaralandı'
    ]
    chunk_size = 2
    max_segmentation_length = 10
    # for i in range(0, len(corpus), chunk_size):
    #     corpus_chunk = set(itertools.chain(*map(morphomize, corpus[i:i + chunk_size])))
    #     if len(corpus_chunk):
    #         segmentation_length = len(max(corpus_chunk, key=lambda l: len(l[1]))[1])
    #         if segmentation_length > max_segmentation_length:
    #             max_segmentation_length = segmentation_length
    #         print(f"processed {i + chunk_size}/{len(corpus)} ")
    #         save_raw_data(corpus_chunk)

    save_structured_data(max_segmentation_length, read_file="raw_corpus")
