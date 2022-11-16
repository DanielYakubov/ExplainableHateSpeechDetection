import torch
import numpy as np
import logging
from tqdm.auto import tqdm

from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader

import evaluate
import sklearn
from transformers import AutoTokenizer, BertModel, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_scheduler
from data.helpers.make_dataset import create_cls_dataset

logging.basicConfig(level=logging.INFO)

TOKENIZER = AutoTokenizer.from_pretrained('bert-base-uncased')
MODEL = BertModel.from_pretrained('bert-base-uncased',
                                 output_hidden_states=True)
                    
# MODEL.eval()

LABEL_TO_NUMBER = {
    "normal": [0, 0],
    "offensive": [1, 0],
    "hatespeech": [0, 1]
}

def get_embeddings(span):
    span  = "[CLS]" + span + "[SEP]"
    tokenized_span = list(TOKENIZER(span))
    indexed_tokens = TOKENIZER.convert_tokens_to_ids(tokenized_span)
    segment_ids = [1] * len(indexed_tokens)
    tok_tensor = torch.tensor([indexed_tokens])
    seg_tensor = torch.tensor([segment_ids])
    with torch.no_grad():
        outputs = MODEL(tok_tensor, seg_tensor)    
        hidden_states = outputs[2]
    token_vecs = hidden_states[-2][0]
    # calculate average of second to last
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding


def compute_metrics(eval_pred):
    metric = evaluate.load("f1") # unbalanced dataset
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predicitions=predictions, references=labels)

def preprocess_dataset(dataset):

    def _tokenize_function(datapoint):
        return TOKENIZER(datapoint["text"], padding="max_length", truncation=True)

    for idx in range(len(dataset)):
        tokenized_span = _tokenize_function(dataset[idx])
        dataset.update(idx, "label", LABEL_TO_NUMBER[dataset[idx]["label"]])
        if dataset[idx]["a_s"] == 0 and dataset[idx]["label"] != [0, 0]:
            dataset.update(idx, "label", [0,0]) # a span is normal if it has no a_s, i.e. it is not a rationale for offensive or hate speech
        dataset.update(idx, "text", tokenized_span)
    for idx in range(len(dataset)):
        print(dataset[idx])
    return dataset


def fine_tune_classifier(model_path_or_name: str, train_dataset, eval_dataset):
    classifier = AutoModelForSequenceClassification.from_pretrained(model_path_or_name, 
                                                                num_labels=5)
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(
        model=classifier,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    return classifier


if __name__ == "__main__":
    # sentence = "this is a sentence to test"
    # v = get_embeddings(sentence)
    # train_dataset = create_cls_dataset('data/datasets/span_annotation_train.tsv')
    # val_dataset = create_cls_dataset('data/datasets/span_annotation_val.tsv')
    # tokenized_train = preprocess_dataset(train_dataset)
    # tokenized_val = preprocess_dataset(val_dataset)
    # classifier = fine_tune_classifier('bert-base-cased', tokenized_train, tokenized_val)
    dataset = load_dataset("yelp_review_full")
    
    def tokenize_func(examples):
        return TOKENIZER(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_func, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(50))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(50))

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=6)
    test_dataloader = DataLoader(small_eval_dataset, batch_size=6)

    classifier = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                                num_labels=5)

    optimizer = AdamW(classifier.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    classifier.to(device)

    # training loop
    progress_bar = tqdm(range(num_training_steps))
    classifier.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = classifier(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        # saving the model per epoch
        torch.save(classifier, 'models/span_classifier_model.pth')
    
    # evaluate
    metric = evaluate.load("f1")
    classifier.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = classifier(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    res = metric.compute(average='micro')
    print(res)

            