from transformers import AutoTokenizer
import datasets
import numpy as np
import seqio
import jax
from transformer import tasks

def extract_dpr_examples(element, tokenizer):
    neig = element["neig"]
    targets = element["targets"]
    chunk_id_list ,candidate_idx_list, candidate_rank_list = neig.reshape([-1,3]).T
    chunks = targets.reshape([-1,64])
    examples_dict = dict()
    for chunk_id,candidate_idx,candidate_rank in zip(chunk_id_list ,candidate_idx_list, candidate_rank_list):
        if chunk_id not in examples_dict:
            examples_dict[chunk_id] = {"question":tokenizer.decode(chunks[chunk_id]), "positive_ctxs":[], "hard_negative_ctxs":[]}
        if candidate_rank<3:
            examples_dict[chunk_id]["positive_ctxs"].append({"text":tokenizer.decode(chunks[candidate_idx])})
        if candidate_rank>15:
            examples_dict[chunk_id]["hard_negative_ctxs"].append({"text":tokenizer.decode(chunks[candidate_idx])})
    yield from list(examples_dict.values())

def get_dataset(name, split):
    def gen():
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        task = seqio.get_mixture_or_task(f"{name}neox_retro_nn20_f20_entirebook_qa_seq1024_16384_wtokens")
        train_set = task.get_dataset(split=split,
                                    sequence_length=None,
                                    shard_info=seqio.ShardInfo(jax.process_index(),jax.process_count()))
        examples = iter(train_set.as_numpy_iterator())
        for x in examples:
            for y in extract_dpr_examples(x, tokenizer):
                yield y
    dataset = list(gen())
    dataset = datasets.Dataset.from_list(gen)
    dataset = dataset.shuffle(seed=42)
    return dataset