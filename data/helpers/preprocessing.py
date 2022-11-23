import ast
import json
import csv
import logging
import urllib.request
from typing import Iterable, List
import numpy as np

from collections import Counter

import benepar
import nltk

logging.basicConfig(level=logging.INFO)


def get_data(url: str) -> str:
    """scrape data from source"""
    data = urllib.request.urlopen(url).read().decode("utf-8")
    return data


def majority_vote(annotators, annotation_point):
    """calculates a ground truth from different annotators using majority voting"""
    if isinstance(annotators[0][annotation_point], str):
            annotator_label_counts = Counter(
                [annotator[annotation_point] for annotator in annotators]
            )
    elif isinstance(annotators[0][annotation_point], list):
        # target is stored in a list, despite there only being one target per post in the dataset
        annotator_label_counts = Counter(
            [
                annotator[annotation_point][0]
                for annotator in annotators
                if annotator[annotation_point]
            ]
        )
    mc_label, mc_count = annotator_label_counts.most_common(1)[0]
    if mc_count >= 2:
        return mc_label


def vector_mean(vectors):
    vec = np.mean(vectors, axis=0)
    # maybe some normalization would be good? Author used sigmoid
    return vec


def get_relevant_data(data, output):
    """Preprocesses data according to the same procedure used by original authors"""
    with open(output, "w") as dump:
        writer = csv.writer(dump, delimiter="\t")
        writer.writerow(
            [
                "post_id",
                "post_tokens",
                "hs_label",
                "rationales"
            ]
        )
        for post_id, entry in data.items():
            # if the text is considered as hate speech, or offensive by majority of the annotators then spans marked
            annotator_hs_label = majority_vote(entry["annotators"], "label")
            if annotator_hs_label in ['offensive', 'hatespeech']:
                annotator_hs_label = 'toxic'
                avg_vec = vector_mean(entry["rationales"])
                avg_vec = avg_vec.tolist()
            elif annotator_hs_label == 'normal':
                annotator_hs_label = "non-toxic"
                avg_vec = [0] * len(entry["post_tokens"])
            else:
                # this is the case of no agreement
                continue
            writer.writerow(
                [
                    post_id,
                    entry["post_tokens"],
                    annotator_hs_label,
                    avg_vec,
                ]
            )

def _get_largest_constituents(tree: nltk.Tree):
    """
    recursive top down tree traversal function that returns constituents top down until the largest
    contituents that are of a uniform case (uppercase, lowercase)
    :param tree: an nltk tree containing some parts that are all capitilized and some parts that are lowercased
    :return: after hitting the base cases, this returns a list of spans
    """
    for child in tree:
        if isinstance(child, nltk.Tree):
            words = child.leaves()
            if all([word.isupper() for word in words]) or all([word.islower() for word in words]):
                yield words
            else:
                yield words
                yield from _get_largest_constituents(child)


def _check_uniform_span(span: List[str], stops: Iterable[str]) -> bool:
    """
    helper function to check uniformity of a span while ignore stopwords
    :param span: a list of str tokens
    :param stops: a stopword list
    :return: True if the span is uniform upper case, False if it consists of only stops or has non-uppercased strs
    """
    lst = [word.isupper() for word in span if word.lower() not in stops]
    return all(lst) if lst else False


def _checked_mixed_span(span, stops) -> bool:
    lst = [word.isupper() for word in span if word.lower() not in stops]
    return any(lst) if lst else False


def get_span_intensity(num: float) -> int:
    """
    condenses float labels in the dataset into 6 int labels
    :param num: a float representing rationale agreeableness
    :return: an int from 0-5
    """
    if num == 0:
        return 0
    elif 0 < num <= 0.2:
        return 1
    elif 0.2 < num <= 0.4:
        return 2
    elif 0.4 < num <= 0.6:
        return 3
    elif 0.6 < num <= 0.8:
        return 4
    else:
        return 5


def write_span_file(input_file: str, output_file: str) -> None:
    """
    writes a file with spans and span labels
    :param input_file: a file path of a csv containing preprocessed data
    :param output_file: a file path for the output csv
    :return: None
    """
    parser = benepar.Parser("benepar_en3")
    stops = set(nltk.corpus.stopwords.words('english'))
    with open(input_file, 'r') as data:
        with open(output_file, 'w') as dump:
            csv_reader = csv.reader(data, delimiter='\t')
            csv_writer = csv.writer(dump, delimiter='\t')
            csv_writer.writerow(['post_id', 'span', 'span_label', 'post_hs_label'])
            next(csv_reader) # we don't care about the header'

            logging.info("beginning iteration through data")
            for id, post_toks, hs_label, rationales in csv_reader:
                post_toks = ast.literal_eval(post_toks)
                rationales = ast.literal_eval(rationales)
                hs = [w.upper() if i > 0 else w for w, i in zip(post_toks, rationales)]
                tree = parser.parse(hs)
                spans = _get_largest_constituents(tree)
                for span in spans:
                    try:
                        start_idx = post_toks.index(span[0].lower())
                        s = slice(start_idx, start_idx + len(span))
                    except ValueError:
                        print(span[0], post_toks)
                        continue
                    lc_span = ' '.join([w.lower() for w in span])
                    if _check_uniform_span(span, stops):
                        attn_vec = rationales[s] # the attention vector that corresponds to the span
                        span_no_stops = [w for w in span if w not in stops] # stops aren't really used as rationales, they were included for constituency parsing
                        intensity_label = get_span_intensity(sum(attn_vec)/len(span_no_stops))
                        csv_writer.writerow([id, lc_span, intensity_label, 'toxic'])
                    elif _checked_mixed_span(span, stops):
                        csv_writer.writerow([id, lc_span, -1, 'mixed']) # -1 is a special label to mean that the contains toxic and non-toxic speech
                    else:
                        csv_writer.writerow([id, lc_span, 0, 'non-toxic'])

        logging.info(f"{output_file} written successfully")


if __name__ == "__main__":
    # publicly available dataset
    data = "../datasets/dataset.json"
    with open(data) as d:
        data = json.load(d)

    cleaned_data = get_relevant_data(data, "../datasets/preprocessed_data.tsv")
    input_file = '../datasets/preprocessed_data.tsv'
    output_file = '../datasets/span_annotation.tsv'
    write_span_file(input_file, output_file)
