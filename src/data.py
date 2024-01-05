from transformers import AutoTokenizer
import datasets
import numpy as np
import seqio
import jax
from transformer import tasks
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
def extract_dpr_examples(element, tokenizer):
    neig = element["neig"]
    targets = element["targets"]
    chunk_id_list ,candidate_idx_list, candidate_rank_list = neig.reshape([-1,3]).T
    chunks = targets.reshape([-1,64])
    chunks = tokenizer.batch_decode(chunks, skip_special_tokens=True)
    examples_dict = dict()
    for chunk_id,candidate_idx,candidate_rank in zip(chunk_id_list ,candidate_idx_list, candidate_rank_list):
        if chunk_id not in examples_dict:
            examples_dict[chunk_id] = {"question":chunks[chunk_id], "positive_ctxs":[], "hard_negative_ctxs":[]}
        if candidate_rank<3:
            examples_dict[chunk_id]["positive_ctxs"].append({"text":chunks[candidate_idx]})
        if candidate_rank>13:
            examples_dict[chunk_id]["hard_negative_ctxs"].append({"text":chunks[candidate_idx]})
    return list(examples_dict.values())

def get_dataset(name:str, split:str):
    shard_id = jax.process_index()
    num_shards=jax.process_count()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    task = seqio.get_mixture_or_task(f"{name}neox_retro_nn20_f20_entirebook_qa_seq1024_16384_wtokens")
    train_set = task.get_dataset(split=split,
                                sequence_length=None,
                                shard_info=seqio.ShardInfo(shard_id,num_shards))
    if split!="train":
        train_set = train_set.shard(10,0)
    examples = list(tqdm(train_set.as_numpy_iterator(),desc="Loading examples"))
    extract_dpr_examples_w_tok =  partial(extract_dpr_examples, tokenizer=tokenizer)
    with Pool(300) as p:
        examples = list(tqdm(p.imap(extract_dpr_examples_w_tok, examples), total=len(examples), desc="Extracting examples"))
    
    gen = sum(tqdm(examples,desc="Summing examples"), [])
    dataset = datasets.Dataset.from_list(gen)
    dataset = dataset.shuffle(seed=42)
    dataset.save_to_disk(f"gs://meliad2_us2/datasets/dpr_datasets/{name}_one_tenth/{split}/hfformat_{shard_id}-{num_shards}")
    
    # return dataset


import fire

if __name__ == "__main__":
    fire.Fire(get_dataset)