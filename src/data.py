from transformers import AutoTokenizer
import datasets
import numpy as np
import seqio
import jax

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial





from more_itertools import peekable



def extract_dpr_examples(element):
    neig = element["neig"]
    targets = element["targets"]
    chunk_id_list ,candidate_idx_list, candidate_rank_list = neig.reshape([-1,3]).T
    chunks = targets.reshape([-1,64])
    examples_dict = dict()
    
    for chunk_id,candidate_idx,candidate_rank in zip(chunk_id_list ,candidate_idx_list, candidate_rank_list):
        if chunk_id not in examples_dict:
            examples_dict[chunk_id] = {"question":chunks[chunk_id], "positive_ctxs":[], "hard_negative_ctxs":[]}
        if candidate_rank==0:
            examples_dict[chunk_id]["positive_ctxs"].append({"text":chunks[candidate_idx]})
        if candidate_rank==5:
            examples_dict[chunk_id]["hard_negative_ctxs"].append({"text":chunks[candidate_idx]})
    final_list = []
    for value in examples_dict.values():
        if len(value["positive_ctxs"])==1 and len(value["hard_negative_ctxs"])==1:
            final_list.append(value)
    return final_list
        

import jax
from functools import partial
from src.proc_utils import run_mapping_pipeline,delayed


import numpy as np
import itertools
from more_itertools import chunked
def unstack_element(element,n_examples=None):
    keys = list(element.keys())
    if n_examples is None:
        n_examples = len(element[keys[0]])
    for i in range(n_examples):
        micro_element = {}
        for key in keys:
            try:
                micro_element[key] = element[key][i]
            except:
                print([(key,len(element[key])) for key in keys])
                raise
        yield micro_element

def parse_psg(p):
    return "Passage: "+ p['title'] + " " + p['text']
def detok_parse_psg(p,tokenizer):
    return "Passage: "+ tokenizer.decode(p['text'])
from functools import wraps
def tokenize_examples(example,
                    tokenizer,
                    q_max_len,
                    p_max_len,
                    query_field="question",
                    pos_field="positive_ctxs",
                    neg_field="hard_negative_ctxs",
                    detokenizer=None,
                    ):
    tokenize = partial(tokenizer,
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        padding=True,
                        truncation=True)
    if detokenizer is None:
        query = "Question: "+str(example[query_field])
        pos_psgs = [parse_psg(p) for p in list(unstack_element(example[pos_field]))]
        neg_psgs = [parse_psg(p) for p in list(unstack_element(example[neg_field]))]
    else:
        query = detokenizer.decode(example[query_field])
        query = "Question: "+str(query)
        pos_psgs = [detok_parse_psg(p, detokenizer) for p in example[pos_field]]
        neg_psgs = [detok_parse_psg(p, detokenizer) for p in example[neg_field]]
    def tok(x,l):
        return dict(tokenize(x, max_length=l,padding='max_length', return_tensors='np'))
    _query = tok(query, q_max_len)
    query_input_ids = _query["input_ids"]
    query_attention_mask = _query["attention_mask"]
    _pos_psgs = [tok(x,p_max_len) for x in pos_psgs ]
    _neg_psgs = [tok(x,p_max_len) for x in neg_psgs ]
    return dict(query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                pos_psgs_input_ids=np.stack([x["input_ids"] for x in _pos_psgs]),
                pos_psgs_attention_mask=np.stack([x["attention_mask"] for x in _pos_psgs]),
                neg_psgs_input_ids=np.stack([x["input_ids"] for x in _neg_psgs]),
                neg_psgs_attention_mask=np.stack([x["attention_mask"] for x in _neg_psgs]),
                )
def create_tokenize_examples(model_args, data_args):
    detokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                                              cache_dir=model_args.cache_dir)
    @wraps(tokenize_examples)
    def our_tokenize_examples(example):
        return [tokenize_examples(example,
                                tokenizer,
                                data_args.q_max_len,
                                data_args.p_max_len,
                                detokenizer=detokenizer)]
    return our_tokenize_examples


    
def shuffled_streaming_iterator(iterable, chunk_size=10, seed=None):
    """
    An iterator that shuffles elements of the given iterable in chunks,
    using a numpy RandomState for reproducibility.
    """
    iterator = iter(iterable)
    random_state = np.random.RandomState(seed)

    for chunk in chunked(iterator, chunk_size):
        random_state.shuffle(chunk)
        for item in chunk:
            yield item


import random
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
class IterableDatasetWrapper(IterableDataset):
    def __init__(self, dataset,split):
        super(IterableDatasetWrapper).__init__()
        self.dataset = dataset
        self.split=split
    def __iter__(self):
        print(f"{self.dataset=}")
        print(f"{type(self.dataset)=}")
        itr = iter(self.dataset())
        if self.split=="train":
            itr =  shuffled_streaming_iterator(itr, chunk_size=1000, seed=42)
            itr =  shuffled_streaming_iterator(itr, chunk_size=20000, seed=43)
        yield from itr
from einops import rearrange
from flax.training.common_utils import shard
def package(result):
    keys = list(result[0].keys())
    batch = {}
    for key in keys:
        try:
            arr = np.array([res[key] for res in result]).squeeze(-2)
            arr = shard(arr)
            if key in ["psgs_input_ids","psgs_attention_mask"]:
                arr = rearrange(arr,'b p n ... -> b (p n) ...')
            batch[key] = arr
        except ValueError:
            print([np.array(res[key]).shape for res in result])
            raise
    query = {"input_ids":batch['query_input_ids'],"attention_mask":batch['query_attention_mask']}
    psgs = {"input_ids":batch['psgs_input_ids'],"attention_mask":batch['psgs_attention_mask']}
    return query,psgs



def format_example(x, n_passages=2, top_elements=1):
    neg_psgs_input_ids = x["neg_psgs_input_ids"]
    neg_psgs_attention_mask = x["neg_psgs_attention_mask"]
    if len(neg_psgs_input_ids)<(n_passages-1):
        return None
    neg_cand_idxs = list(range(len(neg_psgs_input_ids)))
    random.shuffle(neg_cand_idxs)
    neg_idx = neg_cand_idxs[:n_passages-1]
    neg_psgs_input_ids = [neg_psgs_input_ids[i] for i in neg_idx]
    neg_psgs_attention_mask = [neg_psgs_attention_mask[i] for i in neg_idx]
    pos_psgs_input_ids = x["pos_psgs_input_ids"][:top_elements]
    pos_psgs_attention_mask = x["pos_psgs_attention_mask"][:top_elements]
    pos_cand_idxs = list(range(len(pos_psgs_input_ids)))
    random.shuffle(pos_cand_idxs)
    pos_idx = pos_cand_idxs[0]
    pos_psgs_input_ids = pos_psgs_input_ids[pos_idx]
    pos_psgs_attention_mask = pos_psgs_attention_mask[pos_idx]
    psgs_input_ids = np.array([pos_psgs_input_ids] + neg_psgs_input_ids)
    psgs_attention_mask = np.array([pos_psgs_attention_mask] + neg_psgs_attention_mask)
    
    el = dict(query_input_ids=x["query_input_ids"],query_attention_mask=x["query_attention_mask"],
                psgs_input_ids=psgs_input_ids,psgs_attention_mask=psgs_attention_mask)
    return el

from more_itertools import peekable

def load_from_seqio(name, split):
    from transformer import tasks
    import tensorflow as tf
    import seqio
    print(f"inside load_from_seqio with {name=} and {split=}")
    suffix="seq1024" if name!="pg19" else "twi_seq1024"
    ds_name = f"{name}neox_retro_nn20_f20_entirebook_qa_{suffix}_16384_wtokens"
    task = seqio.get_mixture_or_task(ds_name)
    if split=="train":
        dataset = task.get_dataset(split=split,
                                    sequence_length=None,
                                    shard_info=seqio.ShardInfo(jax.process_index(),jax.process_count()),
                                    shuffle=False,
                                    use_cached=False,
                                    )
    else:
        dataset = task.get_dataset(split=split,
                                    sequence_length=None,
                                    shuffle=False
                                    ).take(100)
    print("before fetching")
    itr = dataset.prefetch(tf.data.experimental.AUTOTUNE).as_numpy_iterator()
    if split=="validation":
        itr = list(tqdm(itr,desc="Loading examples from dev"))
    
    return itr




import time
def get_dataloader(split, batch_size, model_args, data_args):

    def create_ds():
        return load_from_seqio(name=data_args.dataset_name,split=split)
    map_functions = [extract_dpr_examples, 
                    create_tokenize_examples(model_args, data_args),
                    lambda x: [format_example(x)]]
    data_stream = run_mapping_pipeline(create_ds,
                                    map_functions=map_functions,
                                    num_workers=20,
                                    maxsize=[1000,1000*256,1000*256, 1000*256],
                                    )
    print("sleeping")
    time.sleep(10)
    print("waking up")
    iterable = IterableDatasetWrapper(data_stream,split=split) 
    dloader= DataLoader(itertools.cycle(iterable),
                            batch_size=batch_size,
                            collate_fn=lambda v: package(v)
                            )
    dloader = peekable(dloader)
    dloader.peek()
    # dl_iter = repeat(dloader)
    return iter(dloader)

    
    # itr = dataset.as_numpy_iterator()
    # itr.next()
    # # itr = peekable(itr)
    # # itr.peek()
    # if split!="train":
    #     itr = list(tqdm(itr,desc="Loading examples from dev"))
    # for x in itr:
    #     yield x
# def get_dataset_iter(dataset, split, model_args, data_args):
#     while True:
#         # tokenizer = AutoTokenizer.from_pretrained(
#         #     "bert-base-uncased",
#         # )
#         # for i,x in enumerate(tqdm(data_stream)):
#         #     if (i%50000)==0:
#         #         print(tokenizer.decode(x["query_input_ids"].squeeze() ))
#         #         print(tokenizer.decode(x["pos_psgs_input_ids"].squeeze() ))
#         #         print(tokenizer.decode(x["neg_psgs_input_ids"].squeeze() ))
#         #     yield format_example(x)
        
import itertools





import fire

if __name__ == "__main__":
    fire.Fire()