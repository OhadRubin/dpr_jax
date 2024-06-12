import os

os.environ["BUCKET"] = "meliad_eu2"
os.environ["LOGURU_LEVEL"] = "INFO"
# os.chdir("~/dpr_jax")
if "DEBUG" in os.environ:
  #alternative: python -m debugpy --wait-for-client --listen localhost:5678 `which seqio_cache_tasks` arg1 arg2
  os.system('kill -9 $(lsof -t -i tcp:5678)')
  import debugpy
  debugpy.listen(5678)
  print("Waiting for debugger attach")
  debugpy.wait_for_client()
  
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
from flax.training.common_utils import shard





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
import jax
from flax.jax_utils import pad_shard_unpad

def get_fwd_functions(model_path, per_device_batch_size):
    p_model = FlaxAutoModel.from_pretrained(f"{model_path}/passage_encoder")
    q_model = FlaxAutoModel.from_pretrained(f"{model_path}/query_encoder")
    @pad_shard_unpad
    @partial(jax.pmap,axis_name="device")
    def apply_p_model(input_ids,attention_mask):
        return p_model(input_ids,attention_mask)[0][:, 0, :]
    @pad_shard_unpad
    @partial(jax.pmap,axis_name="device")
    def apply_q_model(input_ids,attention_mask):
        return q_model(input_ids,attention_mask)[0][:, 0, :]
    def fwd(batch):
        q_batch,p_batch = batch
        q_states = apply_p_model(**q_batch,min_device_batch=per_device_batch_size)
        p_states = apply_q_model(**p_batch,min_device_batch=per_device_batch_size)
        out = q_states,p_states
        return jax.device_get(out)
    return fwd

from tqdm import tqdm
import faiss
import pickle

def main(per_device_batch_size=4):
    model_path = "/home/ohadr/dpr_jax/v7_n7_dscodeparrot_b20.95_wd0.01_steps100000"
    forward_model = get_fwd_functions(model_path,per_device_batch_size=per_device_batch_size)
    batch_size=per_device_batch_size*jax.local_device_count()
    encode_text = create_encode_text()
    dataset = load_from_seqio("codeparrot","validation",repeat=False,limit=None)
    all_dist = []
    for element in tqdm(dataset,desc="Processing books"):
        ds = process_element(element)
        ds = ds.map(encode_text, batched=True,remove_columns=ds.column_names)
        dloader= DataLoader(ds,
                            batch_size=batch_size,
                            collate_fn=lambda v: package(v)
                            )
        all_states = []
        index = faiss.IndexFlatIP(768)
        for i,batch in enumerate(tqdm(dloader,desc="Processing batches")):
            q_states,p_states = forward_model(batch)
            if i>0:
                all_states.append(index.search(q_states, 20))
            index.add(p_states)
            
        D, I = all_states[0]
        assert D.shape[0]==batch_size
        all_states = [(np.ones_like(D),np.ones_like(I))] + all_states
            # all_states.append()
        D,I = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=0), *all_states)
        
        all_dist.append(I)
        
        
        print(I,I.shape)
    with open("out_codeparrot.pkl","wb") as f:
        pickle.dump(all_dist,f)
    print("Done")


import fire
import flax


if __name__ == '__main__':
    fire.Fire(main)









