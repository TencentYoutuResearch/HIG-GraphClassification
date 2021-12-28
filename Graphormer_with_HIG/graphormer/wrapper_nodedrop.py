# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import torch_geometric.datasets
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos

import numpy as np
import scipy.sparse
import scipy.sparse as sp

def get_adj_matrix(edge_index_fea, N):
    adj = torch.zeros([N, N])
    adj[edge_index_fea[0, :], edge_index_fea[1, :]] = 1
    Asp = scipy.sparse.csr_matrix(adj)
    Asp = Asp + Asp.T.multiply(Asp.T > Asp) - Asp.multiply(Asp.T > Asp)
    Asp = Asp + sp.eye(Asp.shape[0])

    D1_ = np.array(Asp.sum(axis=1))**(-0.5)
    D2_ = np.array(Asp.sum(axis=0))**(-0.5)
    D1_ = sp.diags(D1_[:,0], format='csr')
    D2_ = sp.diags(D2_[0,:], format='csr')
    A_ = Asp.dot(D1_)
    A_ = D2_.dot(A_)
    A_ = sparse_mx_to_torch_sparse_tensor(A_)
    return A_

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)



    #print(edge_index)
    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True
    #adj_re = get_adj_matrix(edge_index, N)

    #Asp = scipy.sparse.csr_matrix(adj)
    #Asp = Asp + Asp.T.multiply(Asp.T > Asp) - Asp.multiply(Asp.T > Asp)
    #Asp = Asp + sp.eye(Asp.shape[0])

    #D1_ = np.array(Asp.sum(axis=1))**(-0.5)
    #D2_ = np.array(Asp.sum(axis=0))**(-0.5)
    #D1_ = sp.diags(D1_[:,0], format='csr')
    #D2_ = sp.diags(D2_[0,:], format='csr')
    #A_ = Asp.dot(D1_)
    #A_ = D2_.dot(A_)
    #A_ = sparse_mx_to_torch_sparse_tensor(A_)

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]
                   ] = convert_to_single_emb(edge_attr) + 1

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    # max_dist = 1
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    # edge_input = algos.gen_edge_input(max_dist, adj.numpy(), attn_edge_type.numpy())
    rel_pos = torch.from_numpy((shortest_path_result)).long()
    # rel_pos = torch.from_numpy((adj.numpy())).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.adj = adj
    #item.edge_index = torch.from_numpy(np.array(edge_index))
    #item.A_ = A_
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.rel_pos = rel_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()

    return item


class MyGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        super(MyGraphPropPredDataset, self).download()

    def process(self):
        super(MyGraphPropPredDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


class MyPygPCQM4MDataset(PygPCQM4MDataset):
    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    def process(self):
        super(MyPygPCQM4MDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)

class MyPygPCQM4Mv2Dataset(PygPCQM4Mv2Dataset):
    def download(self):
        super(MyPygPCQM4Mv2Dataset, self).download()

    def process(self):
        super(MyPygPCQM4Mv2Dataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)

class MyZINCDataset(torch_geometric.datasets.ZINC):
    def download(self):
        super(MyZINCDataset, self).download()

    def process(self):
        super(MyZINCDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)
