import torch
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 创建一个数据集
data = [i for i in range(100)]
dataset = MyDataset(data)

### 参数说明
# - `dataset`：数据集，必须是一个继承自`torch.utils.data.Dataset`的对象。
# - `batch_size`：每个batch的数据量。
# - `shuffle`：是否在每个epoch开始时打乱数据顺序。
# - `sampler`：定义从数据集中提取样本的策略，如果设置了`sampler`，`shuffle`必须为`False`。
# - `batch_sampler`：与`sampler`类似，但一次返回一个batch的索引。
# - `num_workers`：加载数据时使用的子进程数量。`0`表示数据将在主进程中加载。
# - `collate_fn`：如何将一个列表的样本组合成一个batch的数据。
# - `pin_memory`：如果`True`，DataLoader将会在返回之前将tensors拷贝到CUDA的固定内存。
# - `drop_last`：如果`True`，则丢弃不能完整形成一个batch的数据。
# - `timeout`：如果大于`0`，则表示从worker中获取一个batch的最大时间（秒）。
# - `worker_init_fn`：每个worker初始化时调用的函数。

# 创建DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True, num_workers=2)

# 迭代数据
for sample in dataloader:
    print(sample)