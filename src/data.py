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
            
class DatasetWrapper:
    def __init__(self, train_data, group_size, tokenizer, p_max_len):
        self.group_size = group_size
        self.data = train_data
        self.tokenizer = tokenizer
        self.p_max_len = p_max_len

    def __len__(self):
        return len(self.data)

    def get_example(self, i, epoch):
        example = self.data[i]
        q = example['query_input_ids']

        pp = example['pos_psgs_input_ids']
        p = pp[0]

        nn = example['neg_psgs_input_ids']
        off = epoch * (self.group_size - 1) % len(nn)
        nn = nn * 2
        nn = nn[off: off + self.group_size - 1]

        return q, [p] + nn

    def get_batch(self, indices, epoch):
        qq, dd = zip(*[self.get_example(i, epoch) for i in map(int, indices)])
        dd = sum(dd, [])
        return dict(self.tokenizer.pad(qq, max_length=32, padding='max_length', return_tensors='np')), dict(
            self.tokenizer.pad(dd, max_length=self.p_max_len, padding='max_length', return_tensors='np'))