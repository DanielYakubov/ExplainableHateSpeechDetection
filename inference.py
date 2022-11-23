import csv
import torch
import nltk
import benepar
import ast
import pandas as pd

from sklearn.dummy import DummyClassifier
import evaluate
from collections import Counter
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from data.helpers.make_dataset import HateSpanDataset
from trainer import TOKENIZER, preprocess_dataset_for_span_classification, preprocess_dataset_for_e2e_classification
from sklearn.metrics import f1_score, accuracy_score

from typing import List

def predict(text: str, classifier):
    """
    processes the data in the way needed for prediction
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    encoded_text = TOKENIZER.encode_plus(text,
    truncation=True,
    padding='max_length',
    max_length=100,
    return_attention_mask=True,
    return_tensors='pt')
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    output = classifier(input_ids, attention_mask)
    _, prediction = torch.max(output[0], dim=1)
    return prediction[0].item()


def tree_based_classification(tree: nltk.tree, classifier):
    """
    classifies a text from the top of the syntax treee until hitting a base case of a text that is hatespeech, normal, offensive or a tree that cannot get any smaller
    :param tree: an nltk tree containing text tobe classified
    :param classifier: a model that classifies at the span level
    :return: after hitting the base cases, this returns a list of (span, label)
    """
    for child in tree:
        if isinstance(child, nltk.Tree):
            words = child.leaves()
            span = ' '.join(words)
            label = predict(span, classifier)
            if label > 0: #0 means mixed case, which means more recursion
                yield span, label
            else:
                yield from tree_based_classification(child, classifier)
        else:
            span = child
            label = predict(span, classifier)
            if label == 0:
                label = 1 # single token cannot be 'mixed', defaulting to non-toxic
            yield span, label

if __name__ == '__main__':
    # # first, evaling BERT models
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # test_encodings, test_labels = preprocess_dataset_for_span_classification('data/datasets/span_annotation_test.tsv')
    # test_dataset = HateSpanDataset(test_encodings, test_labels)
    # test_dataloader = DataLoader(test_dataset, batch_size=64)

    # # evaluate span classifier
    # classifier = torch.load('models/span_classifier_model.pth')
    # metric1 = evaluate.load("f1")
    # metric2 = evaluate.load("accuracy")
    # classifier.eval()

    # span_progress_bar = tqdm(range(len(test_dataloader)))
    # for batch in test_dataloader:
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     with torch.no_grad():
    #         outputs = classifier(**batch)
    #     logits = outputs.logits
    #     predictions = torch.argmax(logits, dim=-1)
    #     metric1.add_batch(predictions=predictions, references=batch["labels"])
    #     metric2.add_batch(predictions=predictions, references=batch["labels"])
    #     span_progress_bar.update(1)
    # res1 = metric1.compute(average='macro')
    # res2 = metric2.compute()
    # with open('res.txt', 'w') as f:
    #     print('Distilbert Span Classification', res1, res2, file=f)

    # # evaluate E2E model
    # test_encodings, test_labels = preprocess_dataset_for_e2e_classification('data/datasets/preprocessed_data_test.tsv')
    # test_dataset = HateSpanDataset(test_encodings, test_labels)
    # test_dataloader = DataLoader(test_dataset, batch_size=64)

    # E2Eclassifier = torch.load('models/e2e_classifier_model.pth')
    # metric1 = evaluate.load("f1")
    # metric2 = evaluate.load("accuracy")
    # E2Eclassifier.eval()

    # e2e_progress_bar = tqdm(range(len(test_dataloader)))
    # for batch in test_dataloader:
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     with torch.no_grad():
    #         outputs = E2Eclassifier(**batch)
    #     logits = outputs.logits
    #     predictions = torch.argmax(logits, dim=-1)
    #     metric1.add_batch(predictions=predictions, references=batch["labels"])
    #     metric2.add_batch(predictions=predictions, references=batch["labels"])
    #     e2e_progress_bar.update(1)
    # res1 = metric1.compute(average='macro')
    # res2 = metric2.compute()
    # with open('res.txt', 'a') as f:
    #     print('Distilbert E2E Classification', res1, res2, file=f)


    # baseline
    dummy_classifier = DummyClassifier(strategy='most_frequent')
    train_data = pd.read_csv('data/datasets/preprocessed_data_train.tsv', delimiter='\t')
    X, y = train_data['post_tokens'], train_data['hs_label']
    dummy_classifier.fit(X, y)
    
    # span classifier for tree approach
    span_classifier = torch.load('models/span_classifier_model.pth')
    parser = benepar.Parser("benepar_en3")
    data =  pd.read_csv('data/datasets/preprocessed_data_test.tsv', delimiter='\t')
    span_classifier.eval()

    posts = [ast.literal_eval(d) for d in data['post_tokens']]
    true_y = data['hs_label']
    pred_y = []
    dummy_y = dummy_classifier.predict(posts)

    progress_bar = tqdm(range(len(posts)))
    with open('pred_rationals.tsv', 'w') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow(['post_id', 'post_toks', 'rationales', 'label'])
        for id, post in zip(data["post_id"], posts):
            tree = parser.parse(post)
            rationales = []
            non_rationales = []
            for text, label in tree_based_classification(tree, span_classifier):
                if label == 2:
                    rationales.append(text)
                else:
                    non_rationales.append(text)
            post_label = 'toxic' if rationales else 'non-toxic'
            pred_y.append(post_label)
            tsv_writer.writerow([id, post, rationales, post_label])
            progress_bar.update(1)
    
    # metrics
    f1_score_tree = f1_score(true_y, pred_y, average='macro')
    accuracy_score_tree = accuracy_score(true_y, pred_y)
    f1_score_dummy = f1_score(true_y, dummy_y, average='macro')
    accuracy_score_dummy = accuracy_score(true_y, dummy_y)

    with open('res.txt', 'a') as res:
        print(f'Dummy Classifier f1: {f1_score_dummy} accuracy: {accuracy_score_dummy}', file=res)
        print(f'Tree Classifier f1: {f1_score_tree} accuracy: {accuracy_score_tree}', file=res)

