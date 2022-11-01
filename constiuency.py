import benepar
import csv
import nltk
import ast
import logging

from typing import Iterable, List

logging.basicConfig(level=logging.INFO)


def _get_largest_constituents(tree: nltk.Tree):
    """
    recursive top down tree traversal function that ultimately returns spans
    that are all upper-cased or all lower-cased
    :param tree: an nltk tree containing some parts that are all capitilized and some parts that are lowercased
    :return: after hitting the base cases, this returns a list of spans
    """
    for child in tree:
        if isinstance(child, nltk.Tree):
            words = child.leaves()
            if all([word.isupper() for word in words]):
                yield words
            elif all([word.islower() for word in words]):
                yield words
            else:
                yield from _get_largest_constituents(child)


def _check_span(span: List[str], stops: Iterable[str]) -> bool:
    """
    helper function to check uniformity of a span while ignore stopwords
    :param span: a list of str tokens
    :param stops: a stopword list
    :return: True if the span is uniform upper case, False if it consists of only stops or has non-uppercased strs
    """
    lst = [word.isupper() for word in span if word not in stops]
    return all(lst) if lst else False


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
            csv_writer.writerow(['post_id', 'span', 'span_label', 'post_hs_label', 'post_target_label'])
            next(csv_reader) # we don't care about the header
            logging.info("beginning iteration through data")
            for id, post_toks, hs_label, target_label, rationales in csv_reader:
                if hs_label != 'normal':
                    # unfortunately, json.loads doesn't work here...
                    post_toks = ast.literal_eval(post_toks)
                    rationales = ast.literal_eval(rationales)
                    hs = [w.upper() if i > 0 else w for w, i in zip(post_toks, rationales)]
                    tree = parser.parse(hs)
                    spans = _get_largest_constituents(tree)
                    for span in spans:
                        if _check_span(span, stops):
                            lc_span = [w.lower() for w in span]
                            csv_writer.writerow([id, lc_span, 1, hs_label, target_label])
                        else:
                            csv_writer.writerow([id, span, 0, hs_label, target_label])
        logging.info(f"{output_file} written successfully")


if __name__ == "__main__":
    input_file = 'original_data.csv'
    output_file = 'span_annotation.csv'
    write_span_file(input_file, output_file)
