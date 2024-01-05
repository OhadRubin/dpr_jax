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
        if candidate_rank==0:
            examples_dict[chunk_id]["positive_ctxs"].append({"text":chunks[candidate_idx]})
        if candidate_rank==19:
            examples_dict[chunk_id]["hard_negative_ctxs"].append({"text":chunks[candidate_idx]})
    final_list = []
    for value in examples_dict.values():
        if len(value["positive_ctxs"])==1 and len(value["hard_negative_ctxs"])==1:
            final_list.append(value)
            # del examples_dict[chunk_id]
    return final_list

        
# from functools import partial
# from multiprocessing import Process, Queue
# from functools import partial

# # Define the worker function template
# def worker(input_queue, output_queue, process_function, limit=None):
#     cnt=0

#     while True:
#         item = input_queue.get()
#         if item is None:
#             break
        
#         for result in process_function(item):
#             output_queue.put(result)
#         if limit is not None:
#             if cnt>limit:
#                 break
#         cnt+=1
#     output_queue.put(None)  # Signal the next worker to shut down






# def data_generator(method, dataset, pooling_obj, lm_tokenizer,
#                    retriever_tokenizer, K, chunk_length, max_neighbors,
#                    batch_size,
#                    limit_books=None
#                    ):
#     process_object = partial(process_book,method=method, pooling_obj=pooling_obj,
#                              lm_tokenizer=lm_tokenizer, retriever_tokenizer=retriever_tokenizer,K=K)
#     process_example = partial(flatten_element,chunk_length=chunk_length,max_neighbors=max_neighbors)
#     # process_stack = partial(stack_candidates,batch_size=batch_size)
#     num_workers = 2  # Two stages in your pipeline

#     # Create the queues for communication
#     queues = [Queue() for _ in range(num_workers + 1)]
#     output_queue = queues[-1]

#     # Create and start the workers for each stage
#     book_worker = Process(target=worker, args=(queues[0], queues[1], process_object), kwargs=dict(limit=limit_books))
#     flatten_worker = Process(target=worker, args=(queues[1], queues[2], process_example))
#     # stacking_worker = Process(target=worker, args=(queues[2], queues[3], process_stack))
#     book_worker.start()
#     flatten_worker.start()
#     try:
#         # Feed the dataset items to the first queue
#         for obj in enumerate(dataset):
#             queues[0].put(obj)

#         # Signal the first worker to shut down after all items are queued
#         queues[0].put(None)
#         def my_iter():
#             while True:
#                 result = output_queue.get()
#                 if result is None:
#                     break
#                 yield result
#         for batch in stack_candidates(my_iter(), batch_size):
#             yield batch
#         # Collect the final results
        
#     finally:
#         # Wait for all workers to finish
#         book_worker.join()
#         flatten_worker.join()
        
def get_dataset(name:str, split:str):
    shard_id = jax.process_index()
    num_shards=jax.process_count()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    task = seqio.get_mixture_or_task(f"{name}neox_retro_nn20_f20_entirebook_qa_seq1024_16384_wtokens")
    train_set = task.get_dataset(split=split,
                                sequence_length=None,
                                shard_info=seqio.ShardInfo(shard_id,num_shards))
    if split=="train":
        train_set = train_set.shard(15,0)
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