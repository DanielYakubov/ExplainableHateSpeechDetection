import torch
import statistics as stats
import numpy as np
import evaluate
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, TrainingArguments, Trainer

import logging
logging.basicConfig(level=logging.INFO)

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL = BertModel.from_pretrained('bert-base-uncased',
                                 output_hidden_states=True)
                    
MODEL.eval()

def get_embeddings(span):
    span  = "[CLS]" + span + "[SEP]"
    tokenized_span = list(TOKENIZER.tokenize(span))
    indexed_tokens = TOKENIZER.convert_tokens_to_ids(tokenized_span)
    segment_ids = [1] * len(indexed_tokens)
    tok_tensor = torch.tensor([indexed_tokens])
    seg_tensor = torch.tensor([segment_ids])
    with torch.no_grad():
        outputs = MODEL(tok_tensor, seg_tensor)    
        hidden_states = outputs[2]
    token_vecs = hidden_states[-2][0]
    print(len(token_vecs))
    # calculate average of second to last
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy") # what else can I use?
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predicitions=predictions, references=labels)


def tokenize_dataset(dataset):
    def _tokenize_function(datapoint):
        return TOKENIZER.tokenize(datapoint["text"], padding="max_length", truncation=True)
    return dataset.map(_tokenize_function, batched=True)


def fine_tune_classifier(model_path_or_name: str, train_dataset, eval_dataset):
    classifier = BertForSequenceClassification.from_pretrained(model_path_or_name, 
                                                                num_labels=3,
                                                                output_attentions = False,
                                                                output_hidden_states = False)
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(
        model=classifier,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == "__main__":
    sentence = "this is a sentence to test"

    get_embeddings(sentence)
    # fine_tune_classifer('bert-base-uncase', 'data/datasets/span_annotation_train.tsv', 'data/datasets/span_annotation_val.tsv')