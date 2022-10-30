import csv
import json
import urllib.request
from collections import Counter

import numpy as np


def get_data(url: str) -> str:
    data = urllib.request.urlopen(url).read().decode("utf-8")
    return data


def majority_vote(annotators, annotation_point):
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
                "target_label",
                "rationales",
            ]
        )
        for post_id, entry in data.items():
            # if the text is considered as hate speech, or offensive by majority of the annotators then spans marked
            annotator_hs_label = majority_vote(entry["annotators"], "label")
            annotator_target_label = majority_vote(
                entry["annotators"], "target"
            )
            if annotator_hs_label in ["hatespeech", "offensive"]:
                avg_vec = vector_mean(entry["rationales"])
                hs_label, target_label = (
                    annotator_hs_label,
                    annotator_target_label,
                )
                avg_vec = avg_vec.tolist()
            else:
                hs_label, target_label = ("normal", "None")
                avg_vec = [0] * len(entry["post_tokens"])
            writer.writerow(
                [
                    post_id,
                    entry["post_tokens"],
                    hs_label,
                    target_label,
                    avg_vec,
                ]
            )


if __name__ == "__main__":
    # publicly available dataset
    url = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
    data = get_data(url)
    data = json.loads(data)

    cleaned_data = get_relevant_data(data, "original_data.csv")
