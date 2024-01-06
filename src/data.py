from transformers import AutoTokenizer
import datasets
import numpy as np
import seqio
import jax
from transformer import tasks
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
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
        if candidate_rank==19:
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
def inner_create_tokenize_examples(tokenizer_name, q_max_len, p_max_len, cache_dir=None):
    detokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=cache_dir,
    )
    @wraps(tokenize_examples)
    def our_tokenize_examples(example):
        return [tokenize_examples(example,
                                tokenizer,
                                q_max_len,
                                p_max_len,
                                detokenizer=detokenizer)]
    return our_tokenize_examples
def create_tokenize_examples(model_args, data_args):
    return inner_create_tokenize_examples(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                                        data_args.q_max_len,
                                        data_args.p_max_len,
                                        cache_dir=model_args.cache_dir)

    
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
    def __init__(self, dataset):
        super(IterableDatasetWrapper).__init__()
        self.dataset = dataset
    def __iter__(self):
        yield from self.dataset
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
def get_dataloader(data, batch_size):
    
    iterable = IterableDatasetWrapper(data) 
    dloader= DataLoader(iterable,
                            batch_size=batch_size,
                            collate_fn=lambda v: package(v)
                            )
    dloader = peekable(dloader)
    dloader.peek()
    return iter(dloader)

def load_from_seqio(name, split):
    shard_id = jax.process_index()
    num_shards=jax.process_count()
    
    task = seqio.get_mixture_or_task(f"{name}neox_retro_nn20_f20_entirebook_qa_seq1024_16384_wtokens")
    dataset = task.get_dataset(split=split,
                                sequence_length=None,
                                **(dict(shard_info=seqio.ShardInfo(shard_id,num_shards)) if split=="train" else {}))
    yield from dataset.as_numpy_iterator()
    
    
def get_dataset(name:str, split:str):
    delayed_dataset =  delayed(partial(load_from_seqio, name=name,split=split))()
    from tqdm import tqdm
    

    data_stream = run_mapping_pipeline(delayed_dataset, map_functions = [extract_dpr_examples, 
                                                                         inner_create_tokenize_examples("bert-base-uncased", 128, 128)],
                                       num_workers=50 if split=="train" else 10,
                                       maxsize=[100,100*256,100*256])
    if split=="train":
        data_stream =  shuffled_streaming_iterator(data_stream, chunk_size=5000, seed=42)
        data_stream =  shuffled_streaming_iterator(data_stream, chunk_size=10000, seed=43)
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
    )
    for i,x in enumerate(tqdm(data_stream)):
        if (i%10000)==0:
            print(tokenizer.decode(x["query_input_ids"].squeeze() ))
            print(tokenizer.decode(x["pos_psgs_input_ids"].squeeze() ))
            print(tokenizer.decode(x["neg_psgs_input_ids"].squeeze() ))
        yield format_example(x)
        

    
# def get_dataset(name:str, split:str):
#     shard_id = jax.process_index()
#     num_shards=jax.process_count()
#     tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
#     task = seqio.get_mixture_or_task(f"{name}neox_retro_nn20_f20_entirebook_qa_seq1024_16384_wtokens")
#     train_set = task.get_dataset(split=split,
#                                 sequence_length=None,
#                                 shard_info=seqio.ShardInfo(shard_id,num_shards))
#     if split=="train":
#         train_set = train_set.shard(15,0)
#     examples = list(tqdm(train_set.as_numpy_iterator(),desc="Loading examples"))
#     extract_dpr_examples_w_tok =  partial(extract_dpr_examples, tokenizer=tokenizer)
#     with Pool(300) as p:
#         examples = list(tqdm(p.imap(extract_dpr_examples_w_tok, examples), total=len(examples), desc="Extracting examples"))
    
#     gen = sum(tqdm(examples,desc="Summing examples"), [])
#     dataset = datasets.Dataset.from_list(gen)
#     dataset = dataset.shuffle(seed=42)
#     dataset.save_to_disk(f"gs://meliad2_us2/datasets/dpr_datasets/{name}_one_tenth/{split}/hfformat_{shard_id}-{num_shards}")
    
#     # return dataset


import fire

if __name__ == "__main__":
    fire.Fire()