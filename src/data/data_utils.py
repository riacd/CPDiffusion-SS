import torch
from torch_geometric.data import Dataset, download_url, Batch, Data
from torch.utils.data import Dataset as Dataset2
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataListLoader, DataLoader

import sys
import os
import random
import csv
from datetime import datetime
import argparse
from numpy import array, cross, pi, arccos, sqrt
from tqdm import tqdm
from time import time
import numpy as np
import seaborn as sns
import torch.nn.functional as F
# from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from scipy.stats import spearmanr
from numpy import nan


# num_ss = 8
# ss_vocab = {'E': 0, 'L': 1, 'I': 2, 'T': 3, 'H': 4, 'B': 5, 'G': 6, 'S': 7, 'PAD': 8}
# rev_ss_vocab = {v: k for k, v in ss_vocab.items()}  # from index to token





def minmax_normalize(feature, ori_range):
    # normalize to between -1,1, feature can have batch dim, ori_range is global range
    return ((feature - ori_range[0]) / (ori_range[1] - ori_range[0]) * 2) - 1


def minmax_unnormalize(feature, ori_range):
    # normalize to between -1,1
    return (feature + 1) / 2 * (ori_range[1] - ori_range[0]) + ori_range[0]





def get_esm_tokens():
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    # batch_converter = alphabet.get_batch_converter()
    model.eval()

    # with torch.no_grad():
    #     results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    #
    # token_representations = results["representations"][33]



# class Cath(Dataset):
#     """
#     Added block-wise features for each data
#     """
#
#     def __init__(self, list_IDs, baseDIR, transform=None, pre_transform=None, pre_filter=None,
#                  max_aa_len = 600, max_b_aa_num = 100):
#         super().__init__(baseDIR, transform, pre_transform, pre_filter)
#         'Initialization'
#         self.list_IDs = list_IDs
#         self.baseDIR = baseDIR
#         self.max_aa_len = max_aa_len
#         self.max_b_aa_num = max_b_aa_num
#         self.data_base_dir = os.path.join(baseDIR, 'process')
#         # get esm embedding
#         self.esm_embed = torch.load(os.path.join(baseDIR, 'esm_embed.pt')) # [20, 1280]
#
#
#     def len(self):
#         'Denotes the total number of samples'
#         return len(self.list_IDs)
#
#     def get(self, index):
#         'Generates one sample of data'
#         # Select sample
#         ID = self.list_IDs[index]
#         data = torch.load(os.path.join(self.data_base_dir, ID))
#
#         del data['distances']
#         del data['edge_dist']
#
#         aa_seq = data.x[:, :20]
#         aa_seq = torch.argmax(aa_seq, dim=1, keepdim=False)
#
#         b_aa_num = data.b_aa_num
#         b_idx = torch.cumsum(b_aa_num, dim=0)
#         ss_embdes = []
#         for ib in range(b_aa_num.shape[0]):
#             if ib == 0:
#                 idx_l, idx_r = 0, b_idx[ib]
#             else:
#                 idx_l, idx_r = b_idx[ib-1], b_idx[ib]
#
#             ib_aa_seq = aa_seq[idx_l:idx_r]
#
#             embed = self.esm_embed[ib_aa_seq.long()].mean(dim=0, keepdim=True) # [aa_len, 1280] -> [1, 1280]
#
#             # new_aa_idx = torch.tensor([self.alphabet.tok_to_idx[amino_acids_type[idx_o.item()]] for idx_o in ib_aa_seq], dtype=aa_seq.dtype)
#
#             # [1, aa_len, 1280]
#             # embed = self.model(new_aa_idx.unsqueeze(0), repr_layers=[33], return_contacts=False)["representations"][33]
#             # embed = embed.squeeze(0).sum(dim=0, keepdim=True) # [1, 1280]
#
#             ss_embdes.append(embed)
#
#             # ib_aa_seq_pad = torch.cat((
#             #     ib_aa_seq, torch.tensor([aa_vocab['PAD']]*(self.max_b_aa_num - ib_aa_seq.shape[0]))) ).unsqueeze(0)
#             # aa_seq_b.append(ib_aa_seq_pad)
#         ss_embdes = torch.cat(ss_embdes, dim=0) # [ss_len, 1280]
#
#         b_edge_attr = data.b_edge_attr
#         if len(data.b_edge_attr.shape) == 1:
#             b_edge_attr = b_edge_attr.unsqueeze(-1)
#
#         graph = Data(
#             x=ss_embdes,  #  [ss_len, 1280] cat
#             b_pos=data.b_pos,  # [ss_len, 3] real
#             b_edge_index=data.b_edge_index,  # [2, edge] int
#             b_edge_attr=b_edge_attr,  # [edge, d_edge] real (d_edge >=1)
#             b_type=F.one_hot(data.b_type, num_classes=len(struc_2nds_res_alphabet)),  # [ss_len,8] one-hot
#         )
#         return graph





class Cath(Dataset):

    def __init__(self, list_IDs, baseDIR, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(baseDIR, transform, pre_transform, pre_filter)
        'Initialization'
        self.list_IDs = list_IDs
        self.baseDIR = baseDIR
        self.data_base_dir = os.path.join(baseDIR, 'process')

    def len(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def get(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        data = torch.load(os.path.join(self.data_base_dir, ID))
        return data