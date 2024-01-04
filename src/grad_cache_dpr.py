

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

import jax
import jax.numpy as jnp

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
from flax.training.common_utils import get_metrics, shard
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, FlaxAutoModel
from transformers import (
    HfArgumentParser,
    set_seed,
)
import os
from dataclasses import dataclass, field
from typing import Optional, List
from data import IterableTrain,TrainDataset

logger = logging.getLogger(__name__)

@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    dataset_name: str = field(
        default="castorini/mr-tydi", metadata={"help": "huggingface dataset name"}
    )
    config_name: str = field(
        default="english", metadata={"help": "huggingface dataset config name"}
    )
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    q_max_len: int = field(
        default=32,
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
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    def __post_init__(self):
        if self.dataset_name is not None:
            info = self.dataset_name.split('/')
            self.dataset_split = info[-1] if len(info) == 3 else 'train'
            self.dataset_name = "/".join(info[:-1]) if len(info) == 3 else '/'.join(info)
            self.dataset_language = 'default'
            if ':' in self.dataset_name:
                self.dataset_name, self.dataset_language = self.dataset_name.split(':')
        else:
            self.dataset_name = 'json'
            self.dataset_split = 'train'
            self.dataset_language = 'default'
        if self.train_dir is not None:
            if os.path.isdir(self.train_dir):
                files = os.listdir(self.train_dir)
                # change all train directory paths to absolute
                self.train_dir = os.path.join(os.path.abspath(os.getcwd()), self.train_dir)
                self.train_path = [
                    os.path.join(self.train_dir, f)
                    for f in files
                    if f.endswith('jsonl') or f.endswith('json')
                ]
            else:
                self.train_path = [self.train_dir]
        else:
            self.train_path = None




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

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)
    normalize: bool = field(default=False)

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
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})
    seed: int = 42
    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})


    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    logging_steps: float = field(
        default=1,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    



def tree_chunk(tree: Any, n_chunk: int, axis: int = 0) -> Any:
    return jax.tree_map(
        lambda v: v.reshape(v.shape[:axis] + (n_chunk, -1) + v.shape[axis + 1:]),
        tree
    )


def tree_unchunk(tree: Any, axis: int = 0) -> Any:
    return jax.tree_map(
        lambda x: x.reshape(x.shape[:axis] + (-1,) + x.shape[axis + 2:]),
        tree
    )
Array = Any


def grad_with_cache(f, **grad_kwargs):
    def cache_f(params, cache, *args, **kwargs):
        return jnp.sum(f(params, *args, **kwargs) * cache)
    return jax.grad(cache_f, **grad_kwargs)


def encode_scan_fn(f, carry, x):
    return carry, f(**x)


def cache_grad_scan_fn(f, params, acc, x):
    cached_grad, kwargs = x

    def fwd_fn(w):
        return f(params=w, **kwargs)

    chunk_grad = grad_with_cache(fwd_fn)(params, cached_grad)
    acc = jax.tree_multimap(lambda u, v: u + v, acc, chunk_grad)
    return acc, None


def chunk_encode(encode_fn):
    def f(**xx):
        _, hh = jax.lax.scan(partial(encode_scan_fn, encode_fn), 0, xx)
        return hh
    return f


def cache_grad(encode_fn):
    def f(params, grad_accumulator, cached_grad, **xx):
        grads, _ = jax.lax.scan(
            partial(cache_grad_scan_fn, encode_fn, params), grad_accumulator, [cached_grad, xx]
        )
        return grads
    return f


def unchunk_args(axis: int = 0, argnums: Iterable[int] = ()):
    def decorator_unchunk(f):
        def g(*args, **kwargs):
            new_args = list(args)
            for i in argnums:
                new_args[i] = tree_unchunk(args[i], axis)
            return f(*new_args, **kwargs)

        return g

    return decorator_unchunk


def _onehot(labels: chex.Array, num_classes: int) -> chex.Array:
    x = labels[..., None] == jnp.arange(num_classes).reshape((1,) * labels.ndim + (-1,))
    x = lax.select(x, jnp.ones(x.shape), jnp.zeros(x.shape))
    return x.astype(jnp.float32)


def p_contrastive_loss(ss: chex.Array, tt: chex.Array, axis: str = 'device') -> chex.Array:
    per_shard_targets = tt.shape[0]
    per_sample_targets = int(tt.shape[0] / ss.shape[0])
    labels = jnp.arange(0, per_shard_targets, per_sample_targets) + per_shard_targets * lax.axis_index(axis)

    tt = lax.all_gather(tt, axis).reshape((-1, ss.shape[-1]))
    scores = jnp.dot(ss, jnp.transpose(tt))

    return optax.softmax_cross_entropy(scores, _onehot(labels, scores.shape[-1]))

class TiedParams(PyTreeNode):
    params: FrozenDict[str, Any]

    @property
    def q_params(self):
        return self.params

    @property
    def p_params(self):
        return self.params

    @classmethod
    def create(cls, params):
        return cls(params=params)


class DualParams(PyTreeNode):
    params: Tuple[FrozenDict[str, Any], FrozenDict[str, Any]]

    @property
    def q_params(self):
        return self.params[0]

    @property
    def p_params(self):
        return self.params[1]

    @classmethod
    def create(cls, *ps):
        if len(ps) == 1:
            return cls(params=ps*2)
        else:
            p_params, q_params = ps
            return cls(params=[p_params, q_params])


class RetrieverTrainState(TrainState):
    params: Union[TiedParams, DualParams]


def retriever_train_step(state, queries, passages, dropout_rng, axis='device'):
    q_dropout_rng, p_dropout_rng, new_dropout_rng = jax.random.split(dropout_rng, 3)

    def compute_loss(params):
        q_reps = state.apply_fn(**queries, params=params.q_params, dropout_rng=q_dropout_rng, train=True)[0][:, 0, :]
        p_reps = state.apply_fn(**passages, params=params.p_params, dropout_rng=p_dropout_rng, train=True)[0][:, 0, :]
        return jnp.mean(p_contrastive_loss(q_reps, p_reps, axis=axis))

    loss, grad = jax.value_and_grad(compute_loss)(state.params)
    loss, grad = jax.lax.pmean([loss, grad], axis)

    new_state = state.apply_gradients(grads=grad)

    return loss, new_state, new_dropout_rng


def grad_cache_train_step(state, queries, passages, dropout_rng, axis='device', q_n_subbatch=1, p_n_subbatch=1):

    def encode_query(params, **kwargs):
        return state.apply_fn(**kwargs, params=params.q_params, train=True)[0][:, 0, :]

    def encode_passage(params, **kwargs):
        return state.apply_fn(**kwargs, params=params.p_params, train=True)[0][:, 0, :]

    queries, passages = tree_chunk(queries, q_n_subbatch), tree_chunk(passages, p_n_subbatch)
    q_rngs, p_rngs, new_rng = jax.random.split(dropout_rng, 3)
    q_rngs = jax.random.split(q_rngs, q_n_subbatch)
    p_rngs = jax.random.split(p_rngs, p_n_subbatch)

    q_reps = chunk_encode(partial(encode_query, state.params))(**queries, dropout_rng=q_rngs)
    p_reps = chunk_encode(partial(encode_passage, state.params))(**passages, dropout_rng=p_rngs)

    @unchunk_args(axis=0, argnums=(0, 1))
    def compute_loss(xx, yy):
        return jnp.mean(p_contrastive_loss(xx, yy, axis=axis))

    loss, (q_grads, p_grads) = jax.value_and_grad(compute_loss, argnums=(0, 1))(q_reps, p_reps)

    grads = jax.tree_map(lambda v: jnp.zeros_like(v), state.params)
    grads = cache_grad(encode_query)(state.params, grads, q_grads, **queries, dropout_rng=q_rngs)
    grads = cache_grad(encode_passage)(state.params, grads, p_grads, **passages, dropout_rng=p_rngs)

    loss, grads = jax.lax.pmean([loss, grads], axis)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state, new_rng





def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TevatronTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TevatronTrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )


    # model_args = ModelArguments("bert-base-uncased")
    # data_args = DataArguments(dataset_name="castorini/mr-tydi",config_name="english")


    # training_args = TevatronTrainingArguments(output_dir="blabla")



    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    try:
        model = FlaxAutoModel.from_pretrained(
            model_args.model_name_or_path, config=config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )
    except:
        model = FlaxAutoModel.from_pretrained(
            model_args.model_name_or_path, config=config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype),
            from_pt=True
        )

    if data_args.train_dir:
        data_files = {
            'train': data_args.train_path
        }
    else:
        data_files = None

    train_dataset = \
        datasets.load_dataset(data_args.dataset_name, data_args.config_name, cache_dir=model_args.cache_dir,
                                data_files=data_files)[data_args.dataset_split]

    def tokenize_train(example):
        tokenize = partial(tokenizer, return_attention_mask=False, return_token_type_ids=False, padding=False,
                            truncation=True)
        query = example['query']
        pos_psgs = [p['title'] + " " + p['text'] for p in example['positive_passages']]
        neg_psgs = [p['title'] + " " + p['text'] for p in example['negative_passages']]

        example['query_input_ids'] = dict(tokenize(query, max_length=data_args.q_max_len))
        example['pos_psgs_input_ids'] = [dict(tokenize(x, max_length=data_args.p_max_len)) for x in pos_psgs]
        example['neg_psgs_input_ids'] = [dict(tokenize(x, max_length=data_args.p_max_len)) for x in neg_psgs]

        return example

    train_data = train_dataset.map(
        tokenize_train,
        batched=False,
        num_proc=data_args.dataset_proc_num,
        desc="Running tokenizer on train dataset",
    )
    train_data = train_data.filter(
        function=lambda data: len(data["pos_psgs_input_ids"]) >= 1 and \
                                len(data["neg_psgs_input_ids"]) >= data_args.train_n_passages-1, num_proc=64
    )



    train_dataset = TrainDataset(train_data, data_args.train_n_passages, tokenizer)

    def create_learning_rate_fn(
            train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int,
            learning_rate: float
    ):
        """Returns a linear warmup, linear_decay learning rate function."""
        steps_per_epoch = train_ds_size // train_batch_size
        num_train_steps = steps_per_epoch * num_train_epochs
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

    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        int(training_args.num_train_epochs),
        int(total_train_steps * 0.1),
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

    # if model_args.untie_encoder:
    #     params = DualParams.create(model.params)
    # else:
    from collections import namedtuple
    ParamTuple = namedtuple("ParamTuple","params q_params p_params")
    params = ParamTuple(params=model.params,q_params=model.params,p_params=model.params)
    state = RetrieverTrainState.create(apply_fn=model.__call__, params=params, tx=adamw)

    if training_args.grad_cache:
        q_n_subbatch = train_batch_size // training_args.gc_q_chunk_size
        p_n_subbatch = train_batch_size * data_args.train_n_passages // training_args.gc_p_chunk_size
        p_train_step = jax.pmap(
            partial(grad_cache_train_step, q_n_subbatch=q_n_subbatch, p_n_subbatch=p_n_subbatch),
            "device"
        )
    else:
        p_train_step = jax.pmap(
            retriever_train_step,
            "device"
        )

    state = jax_utils.replicate(state)
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())



    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_metrics = []
    for epoch in tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0):
        # ======================== Training ================================
        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        steps_per_epoch = len(train_dataset) // train_batch_size

        batch_idx = jax.random.permutation(input_rng, len(train_dataset))
        batch_idx = batch_idx[: steps_per_epoch * train_batch_size]
        batch_idx = batch_idx.reshape((steps_per_epoch, train_batch_size)).tolist()
        iterable_train = IterableTrain(train_dataset, batch_idx, epoch)
        train_loader = prefetch_to_device(
            iter(DataLoader(iterable_train,
                num_workers=16, prefetch_factor=256, batch_size=None, collate_fn=lambda v: v)
            ), 2)

        # train
        epochs = tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False)
        for step in epochs:
            cur_step = epoch * (len(train_dataset) // train_batch_size) + step
            batch = next(train_loader)

            loss, state, dropout_rngs = p_train_step(state, *batch, dropout_rngs)
            train_metrics.append({'loss': loss})

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                train_metrics = get_metrics(train_metrics)
                print(
                    f"Step... ({cur_step} | Loss: {train_metrics['loss'].mean()},"
                    f" Learning Rate: {linear_decay_lr_schedule_fn(cur_step)})",
                    flush=True,
                )
                train_metrics = []

        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs})"
        )

    params = jax_utils.unreplicate(state.params)

    if model_args.untie_encoder:
        os.makedirs(training_args.output_dir, exist_ok=True)
        model.save_pretrained(os.path.join(training_args.output_dir, 'query_encoder'), params=params.q_params)
        model.save_pretrained(os.path.join(training_args.output_dir, 'passage_encoder'), params=params.p_params)
    else:
        model.save_pretrained(training_args.output_dir, params=params.p_params)


if __name__ == "__main__":
    main()