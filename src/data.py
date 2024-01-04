from flax.training.common_utils import get_metrics, shard
from torch.utils.data import DataLoader, IterableDataset
class IterableTrain(IterableDataset):
    def __init__(self, dataset, batch_idx, epoch):
        super(IterableTrain).__init__()
        self.dataset = dataset
        self.batch_idx = batch_idx
        self.epoch = epoch

    def __iter__(self):
        for idx in self.batch_idx:
            batch = self.dataset.get_batch(idx, self.epoch)
            batch = shard(batch)
            yield batch
            
def get_example(i, epoch, group_size, data):
    example = data[i]
    q = example['query_input_ids']

    pp = example['pos_psgs_input_ids']
    p = pp[0]

    nn = example['neg_psgs_input_ids']
    off = epoch * (group_size - 1) % len(nn)
    nn = nn * 2
    nn = nn[off: off + group_size - 1]

    return q, [p] + nn


def get_batch(indices, epoch, p_max_len, tokenizer, group_size, data):
    qq, dd = zip(*[get_example(i, epoch, group_size, data) for i in map(int, indices)])
    dd = sum(dd, [])
    return dict(tokenizer.pad(qq, max_length=32, padding='max_length', return_tensors='np')), dict(
        tokenizer.pad(dd, max_length=p_max_len, padding='max_length', return_tensors='np'))
    
    
class DatasetWrapper:
    def __init__(self, train_data, group_size, tokenizer, p_max_len):
        self.group_size = group_size
        self.data = train_data
        self.tokenizer = tokenizer
        self.p_max_len = p_max_len
    def __len__(self):
        return len(self.data)
    
    def get_batch(self, indices, epoch):
        return get_batch(indices, epoch, self.p_max_len, self.tokenizer, self.group_size, self.data)


