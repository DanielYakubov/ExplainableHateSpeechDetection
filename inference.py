import torch
import nltk
import benepar
import ast
import pandas as pd

from collections import Counter
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from data.helpers.make_dataset import HateSpanDataset
from trainer import TOKENIZER

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
    classifier = torch.load('models/span_classifier_model.pth')
    parser = benepar.Parser("benepar_en3")
    
    file = 'data/datasets/preprocessed_data_val.tsv'
    data =  pd.read_csv(file, delimiter='\t')
    posts = [ast.literal_eval(d) for d in data['post_tokens']]
    classifier.eval()

    true = data['hs_label']
    true_bal = [(k, v/len(true)*100) for k, v in Counter(true).items()]
    print(true_bal)
    correct = 0

    progress_bar = tqdm(range(len(posts)))

    for post, gold in zip(posts, true):
        tree = parser.parse(post)
        rationales = []
        non_rationales = []
        for text, label in tree_based_classification(tree, classifier):
            if label == 2:
                rationales.append(text)
            else:
                non_rationales.append(text)
        if rationales:
            post_label = 'toxic'
            if non_rationales:
                print()
                print(' '.join(post), "\n--------\n", rationales)
                print()
        else:
            post_label = 'non-toxic'
        if post_label == gold:
            correct+=1
        progress_bar.update(1)
    print('-'*30)
    print(correct/len(true))