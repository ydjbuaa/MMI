import torch
import math
from torch.autograd import Variable
from utils.vocab import Constants, DepTransitions


class Dataset(object):
    def __init__(self, src_data, trg_data, batch_size, cuda, volatile=False, ret_limit=None):
        self.src = src_data

        if trg_data:
            self.trg = trg_data
            assert (len(self.src) == len(self.trg))
            if ret_limit:
                self.src = self.src[:ret_limit]
                self.trg = self.trg[:ret_limit]
        else:
            self.trg = None
            if ret_limit:
                self.src = self.src[:ret_limit]

        self.cuda = cuda

        self.batch_size = batch_size
        self.num_samples = len(self.src)
        self.num_batches = math.ceil(len(self.src) / batch_size)
        self.volatile = volatile

    @staticmethod
    def _batch_identity(data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.num_batches, "%d > %d" % (index, self.num_batches)
        src_batch, lengths = self._batch_identity(
            self.src[index * self.batch_size:(index + 1) * self.batch_size],
            align_right=False, include_lengths=True)

        if self.trg:
            trg_batch, _ = self._batch_identity(
                self.trg[index * self.batch_size:(index + 1) * self.batch_size],
                align_right=False, include_lengths=True)
        else:
            trg_batch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(src_batch))
        batch = (zip(indices, src_batch) if trg_batch is None
                 else zip(indices, src_batch, trg_batch))

        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if trg_batch is None:
            indices, src_batch = zip(*batch)
        else:
            indices, src_batch, trg_batch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0)
            b = b.t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)
        lengths = Variable(lengths, volatile=self.volatile)
        return (wrap(src_batch), lengths), \
               wrap(trg_batch), indices

    def __len__(self):
        return self.num_batches

    def shuffle(self):
        data = list(zip(self.src, self.trg))
        self.src, self.trg = zip(*[data[i] for i in torch.randperm(len(data))])


class DepTreeDataset(object):
    def __init__(self, src_data, src_tree, trg_data, trg_tree, batch_size, cuda, volatile=False):
        self.src = src_data
        self.src_tree = src_tree

        assert len(self.src) == len(self.src_tree)

        if trg_data:
            self.trg = trg_data
            self.trg_tree = trg_tree
            assert (len(self.src) == len(self.trg))

        else:
            self.trg = None

        self.cuda_flag = cuda
        self.batch_size = batch_size
        self.num_samples = len(self.src)
        self.num_batches = math.ceil(len(self.src) / batch_size)
        self.volatile = volatile

    @staticmethod
    def _batch_identity(data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        if include_lengths:
            return out, lengths
        else:
            return out

    @staticmethod
    def _batch_syntax_identity(data, syn_indices):
        lengths = [x.size(0) for x in data]
        batch_size = len(lengths)
        max_length = max(lengths)

        out = data[0].new(len(data), max_length).fill_(Constants.PAD)
        indices1 = torch.LongTensor(list(range(max_length)) * batch_size).view(batch_size, max_length)
        indices2 = torch.LongTensor(list(range(max_length)) * batch_size).view(batch_size, max_length)
        indices3 = torch.LongTensor(list(range(max_length)) * batch_size).view(batch_size, max_length)
        indices4 = torch.LongTensor(list(range(max_length)) * batch_size).view(batch_size, max_length)

        for i in range(len(data)):
            data_length = data[i].size(0)
            out[i].narrow(0, 0, data_length).copy_(data[i])
            indices1[i].narrow(0, 0, data_length).copy_(syn_indices[i][0])
            indices2[i].narrow(0, 0, data_length).copy_(syn_indices[i][1])
            indices3[i].narrow(0, 0, data_length).copy_(syn_indices[i][2])
            indices4[i].narrow(0, 0, data_length).copy_(syn_indices[i][3])

        return out, lengths, [indices1, indices2, indices3, indices4]

    def __getitem__(self, index):
        assert index < self.num_batches, "%d > %d" % (index, self.num_batches)
        src_batch, lengths, syn_indices = self._batch_syntax_identity(
            self.src[index * self.batch_size:(index + 1) * self.batch_size],
            self.src_tree[index * self.batch_size:(index + 1) * self.batch_size])

        if self.trg:
            trg_batch = self._batch_identity(
                self.trg[index * self.batch_size:(index + 1) * self.batch_size],
                align_right=False, include_lengths=False)
        else:
            trg_batch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(src_batch))
        batch = (
        zip(indices, src_batch, syn_indices[0], syn_indices[1], syn_indices[2], syn_indices[3]) if trg_batch is None
        else zip(indices, src_batch, syn_indices[0], syn_indices[1], syn_indices[2], syn_indices[3], trg_batch))

        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if trg_batch is None:
            indices, src_batch, syn1, syn2, syn3, syn4 = zip(*batch)
        else:
            indices, src_batch, syn1, syn2, syn3, syn4, trg_batch = zip(*batch)

        def wrap(b, t_flag=False):
            if b is None:
                return b
            b = torch.stack(b, 0)
            if t_flag:
                b = b.t().contiguous()
            if self.cuda_flag:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        return (wrap(src_batch), [wrap(syn1), wrap(syn2), wrap(syn3), wrap(syn4)]), \
                wrap(trg_batch, t_flag=True), indices

    def __len__(self):
        return self.num_batches

# if __name__ == '__main__':
#     train_set = torch.load('../data/stc/train/train.tree.pt')
#     dataset = TreeDataset(train_set['src'], train_set['src_tree'],
#                           train_set['trg'], train_set['trg_tree'],
#                           20, cuda=True, volatile=False)
#     print(dataset.num_samples)
#     src_batch, trg_batch, _ = dataset[0]
#     src_data, src_syn = src_batch
#     print(len(src_syn))
#     print(src_syn[0])
#     print(src_syn[1])
#     print(src_syn[2])
#     print(src_syn[3])
#     print(trg_batch)
#     pass
