import torch
import statistics as stats
from transformers import BertTokenizer, BertModel

import logging
logging.basicConfig(level=logging.INFO)

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL = BertModel.from_pretrained('bert-base-uncased',
                                 output_hidden_states=True)
MODEL.eval()

def get_embeddings(span):
    span  = "[CLS]" + span + "[SEP]"
    tokenized_span = list(TOKENIZER.tokenize(span))
    print(tokenized_span)
    indexed_tokens = TOKENIZER.convert_tokens_to_ids(tokenized_span)
    segment_ids = [1] * len(indexed_tokens)
    tok_tensor = torch.tensor([indexed_tokens])
    seg_tensor = torch.tensor([segment_ids])
    with torch.no_grad():
        outputs = MODEL(tok_tensor, seg_tensor)    
        hidden_states = outputs[2]
    token_vecs = hidden_states[-2][0]
    print(token_vecs)
    # calculate average of second to last
    sentence_embedding = torch.mean(token_vecs, dim=0)
    print(sentence_embedding.size())
    return sentence_embedding

if __name__ == "__main__":
    sentence = "this is a sentence to test"
    get_embeddings(sentence)