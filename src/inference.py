import os

os.environ["BUCKET"] = "meliad_eu2"
os.environ["LOGURU_LEVEL"] = "INFO"
# os.chdir("~/dpr_jax")
from src.data import load_from_seqio
import sys
from transformers import AutoConfig, AutoTokenizer, FlaxAutoModel
sys.path.append("/home/ohadr/meliad2")

from datasets import load_dataset
from functools import partial
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import datasets
from functools import partial
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


dataset = load_from_seqio("codeparrot","validation",repeat=False,limit=None)



# Process the dataset
def process_element(element):
    targets = element["targets"]
    chunks = targets.reshape([-1, 64])
    ds = datasets.Dataset.from_generator(lambda: map(lambda x:dict(input_ids=x),chunks))
    return ds

# from contextutils import wraps
from functools import wraps

def create_encode_text():
    detokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenize = partial(tokenizer,
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        truncation=True,
                        max_length=128,
                        padding='max_length', return_tensors='np')
    def _encode_text(batch):
        input_ids = batch["input_ids"]
        detok_input_ids = detokenizer.batch_decode(input_ids)
        detok_passage = ["Passage: "+x for x in detok_input_ids]
        detok_question = ["Question: "+x for x in detok_input_ids]
        passage_dict = dict(tokenize(detok_passage))
        question_dict = dict(tokenize(detok_question))
        passage_dict = {"psgs_"+k:v for k,v in passage_dict.items()}
        question_dict = {"query_"+k:v for k,v in question_dict.items()}
        batch.update(**passage_dict,**question_dict)
        return batch
    return _encode_text

def package(result):
    keys = list(result[0].keys())
    batch = {}
    for key in keys:
        try:
            arr = np.array([res[key] for res in result])
            batch[key] = arr
        except ValueError:
            print([np.array(res[key]).shape for res in result])
            raise
    query = {"input_ids":batch['query_input_ids'],"attention_mask":batch['query_attention_mask']}
    psgs = {"input_ids":batch['psgs_input_ids'],"attention_mask":batch['psgs_attention_mask']}
    return query,psgs

import numpy as np

def main():
    encode_text = create_encode_text()
    ds = ds.map(encode_text, batched=True,remove_columns=ds.column_names)
    dloader= DataLoader(ds,
                        batch_size=10,
                        collate_fn=lambda v: package(v)
                        )
    batch = next(iter(dloader))
    q_batch,p_batch = batch
    model = FlaxAutoModel.from_pretrained("/home/ohadr/dpr_jax/v7_n7_dscodeparrot_b20.95_wd0.01_steps100000/passage_encoder")
    print(model(**p_batch)[0][:, 0, :])












