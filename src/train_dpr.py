
# import os
# os.system("sudo kill -9 $(sudo lsof -w /dev/accel0 | awk 'NR>1{print $2}' |uniq)")
import os

# os.system("sudo kill -9 $(sudo lsof -w /dev/accel0 | awk 'NR>1{print $2}' |uniq)")
# os.system("sudo kill -9 $(sudo lsof -w /dev/accel1 | awk 'NR>1{print $2}' |uniq)")
# os.system("sudo kill -9 $(sudo lsof -w /dev/accel2 | awk 'NR>1{print $2}' |uniq)")
# os.system("sudo kill -9 $(sudo lsof -w /dev/accel3 | awk 'NR>1{print $2}' |uniq)")


# # os.system('if  pgrep -f -a "ht_main.py" ; then killall -q -w -s SIGKILL ht_main.py ; fi')
# os.system('rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs')
# import time
# time.sleep(5)
import os

if "DEBUG" in os.environ:
  #alternative: python -m debugpy --wait-for-client --listen localhost:5678 `which seqio_cache_tasks` arg1 arg2
  os.system('kill -9 $(lsof -t -i tcp:5678)')
  import debugpy
  debugpy.listen(5678)
  print("Waiting for debugger attach")
  debugpy.wait_for_client()
import jax
print(jax.devices())

import seqio
from transformer import tasks
from functools import partial

import jax
import jax.numpy as jnp


from functools import partial
from typing import Tuple, Any, Union

import jax
from jax import numpy as jnp

from flax.training.train_state import TrainState
from flax.core import FrozenDict
from flax.struct import PyTreeNode

import jax.numpy as jnp
from jax import lax
import optax
import chex

from typing import Iterable, Any
from functools import partial


from typing import Any
import jax
import logging
import os
import sys
from functools import partial

import datasets
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, traverse_util
from flax.jax_utils import prefetch_to_device

from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, FlaxAutoModel
from transformers import (
    HfArgumentParser,
    set_seed,
)
import os
from dataclasses import dataclass, field
from typing import Optional, List
from src.data import get_dataloader, get_dataset_iter
from collections import namedtuple
from datasets import disable_caching
disable_caching()

ParamTuple = namedtuple("ParamTuple","q_params p_params")
logger = logging.getLogger(__name__)

from einops import rearrange
import numpy as np
from tqdm import tqdm
from metric_utils import p_calc_scores, calc_metrics, get_metrics


import numpy as np
import time
import wandb

from fsspec.generic import rsync
import shutil


def save_to_cloud(model, params, remote_model_path, local_model_path = "/tmp/local_model"):
    model.save_pretrained(local_model_path, params=params)
    rsync(local_model_path, remote_model_path)
    shutil.rmtree(local_model_path)

@dataclass
class DataArguments:
    dataset_name: str = field(
        default="codeparrot", metadata={"help": "huggingface dataset name"}
    )
    q_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )



@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )
    
@dataclass
class TevatronTrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    
    warmup_steps: int = field(default=2000)
    seed: int = 42
    num_train_steps: float = field(default=50000, metadata={"help": "Total number of training steps to perform."})
    learning_rate: float = field(default=4e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    per_device_train_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer"
            )
        },
    )
    eval_steps: float = field(default=1000)
    n_eval_steps: int = field(default=100)
    
        
    

Array = Any



class RetrieverTrainState(TrainState):
    params: PyTreeNode
    



def retriever_train_step(state, queries, passages, dropout_rng, axis='device'):
    q_dropout_rng, p_dropout_rng, new_dropout_rng = jax.random.split(dropout_rng, 3)

    def compute_loss(params):
        q_reps = state.apply_fn(**queries, params=params.q_params, dropout_rng=q_dropout_rng, train=True)[0][:, 0, :]
        p_reps = state.apply_fn(**passages, params=params.p_params, dropout_rng=p_dropout_rng, train=True)[0][:, 0, :]
        scores, labels = p_calc_scores(q_reps, p_reps, axis=axis)
        loss, metrics = calc_metrics(scores, labels)
        return loss, metrics

    (loss,metrics), grad = jax.value_and_grad(compute_loss, has_aux=True)(state.params)
    loss, grad, metrics = jax.lax.pmean([loss, grad, metrics], axis)

    new_state = state.apply_gradients(grads=grad)

    return metrics, new_state, new_dropout_rng


def retriever_eval_step(state, queries, passages, dropout_rng, axis='device'):
    q_dropout_rng, p_dropout_rng, new_dropout_rng = jax.random.split(dropout_rng, 3)

    def compute_loss(params):
        q_reps = state.apply_fn(**queries, params=params.q_params, dropout_rng=q_dropout_rng, train=True)[0][:, 0, :]
        p_reps = state.apply_fn(**passages, params=params.p_params, dropout_rng=p_dropout_rng, train=True)[0][:, 0, :]
        scores, labels = p_calc_scores(q_reps, p_reps, axis=axis)
        loss, metrics = calc_metrics(scores, labels)
        return loss, metrics

    _,metrics = compute_loss(state.params)
    metrics = jax.lax.pmean(metrics, axis)
    return metrics, state, new_dropout_rng



def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TevatronTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TevatronTrainingArguments


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )


    num_train_steps = int(training_args.num_train_steps)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.local_device_count()
    
    validation_dataset = get_dataset("validation", data_args)
    validation_data = get_dataset_iter(validation_dataset, "validation", model_args, data_args)
    validation_loader = get_dataloader(validation_data, train_batch_size)
    
    train_dataset = get_dataset("train", data_args)
    train_data = get_dataset_iter(train_dataset, "train", model_args, data_args)
    train_loader = get_dataloader(train_data,train_batch_size)

    try:
        model = FlaxAutoModel.from_pretrained(
            model_args.model_name_or_path, config=config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )
    except:
        model = FlaxAutoModel.from_pretrained(
            model_args.model_name_or_path, config=config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype),
            from_pt=True
        )
    def create_learning_rate_fn(
            num_train_steps:int,
            num_warmup_steps: int,
            learning_rate: float
    ):
        """Returns a linear warmup, linear_decay learning rate function."""
        warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
        decay_fn = optax.linear_schedule(
            init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
        )
        schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
        return schedule_fn

    def _decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        layer_norm_params = [
            (name, "scale") for name in ["self_attn_layer_norm", "layernorm_embedding", "final_layer_norm"]
        ]
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    def decay_mask_fn(params):
        param_nodes, treedef = jax.tree_flatten(params, lambda v: isinstance(v, dict))
        masks = [_decay_mask_fn(param_node) for param_node in param_nodes]
        return jax.tree_unflatten(treedef, masks)


    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        num_train_steps,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        mask=decay_mask_fn,
    )


    if model_args.untie_encoder:
        model_copy = jax.tree_map(jnp.copy, model.params)
        params = ParamTuple(q_params=model.params,p_params=model_copy)
    else:
        params = ParamTuple(q_params=model.params,p_params=model.params)
    state = RetrieverTrainState.create(apply_fn=model.__call__, params=params, tx=adamw)

    p_train_step = jax.pmap(
        retriever_train_step,
        "device"
    )
    p_eval_step = jax.pmap(
        retriever_eval_step,
        "device"
    )

    state = jax_utils.replicate(state)
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())



    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {num_train_steps}")

    train_metrics = []
    is_main = jax.process_index() == 0
    if is_main:
        wandb.init(project="dpr_jax", resume="allow")

    for step in tqdm(range(num_train_steps), position=0):
        # ======================== Training ================================
        batch = next(train_loader)
        metrics, state, dropout_rngs = p_train_step(state, *batch, dropout_rngs)
        train_metrics.append(jax.tree_map(lambda x:x.mean(),metrics))

        if step % training_args.logging_steps == 0 and step > 0:
            train_metrics = get_metrics(train_metrics)
            train_metrics = jax.tree_map(lambda x:x.mean(),train_metrics)
            loss = train_metrics['loss']
            lr = linear_decay_lr_schedule_fn(step)
            print(
                f"Step... ({step} | Loss: {loss},"
                f" Learning Rate: {lr})",
                flush=True,
            )
            if is_main:
                wandb.log({"lr":lr, **{f"train/{k}":v for k,v in train_metrics.items()}})
            
            train_metrics = []
        if step % training_args.eval_steps == 0 and step > 0:
            eval_metrics = []
            for _ in tqdm(range(training_args.n_eval_steps), desc="Evaluating...", position=2, leave=False):
                batch = next(validation_loader)
                metrics, state, dropout_rngs = p_eval_step(state, *batch, dropout_rngs)
                eval_metrics.append(jax.tree_map(lambda x:x.mean(),metrics))
            eval_metrics = get_metrics(eval_metrics)
            eval_metrics = jax.tree_map(lambda x:x.mean(),eval_metrics)
            loss = eval_metrics['loss'].mean()
            
            print(
                f"Eval result: : Step: ({step} | Loss: {loss},",
                flush=True,
            )
            if is_main:
                wandb.log({f"validation/{k}":v for k,v in eval_metrics.items()})
            
    params = jax_utils.unreplicate(state.params)
    if is_main:
        if model_args.untie_encoder:

            save_to_cloud(model,params=params.q_params, remote_model_path=f'{training_args.output_dir}/query_encoder')
            save_to_cloud(model,params=params.p_params, remote_model_path=f'{training_args.output_dir}/passage_encoder')
        else:
            save_to_cloud(model,params=params.p_params, remote_model_path=training_args.output_dir)


if __name__ == "__main__":
    main()