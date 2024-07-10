import random
import os

import torch
import torch.nn.functional as F

# from torch_geometric.utils.sparse import dense_to_sparse
#
#
# from Bio.PDB import PDBParser
# from Bio.PDB.DSSP import DSSP

import argparse

from tqdm import tqdm

from .cath_2nd import aa_vocab, struc_2nds_res_alphabet8, struc_2nds_res_alphabet3, amino_acids_type, ss8_vocab, ss3_vocab

from sklearn.decomposition import PCA

from torch_geometric.data import InMemoryDataset, Data

from .protein_utils import NormalizeProtein, get_stat
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'





def get_store_esm_embeddings(basedir):
    esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    model = esm_model.eval()

    aa_embed = []
    for iaa, aa_type in enumerate(amino_acids_type):
        aa_idx = torch.tensor([alphabet.tok_to_idx[aa_type]]).unsqueeze(0)
        with torch.no_grad():
            embed = model(aa_idx, repr_layers=[33], return_contacts=False)["representations"][33] # [1,1,1280]
        aa_embed.append(embed.squeeze(0))  # [1,1280]
    aa_embed = torch.cat(aa_embed, dim=0) # [20, 1280]

    # use 128 dims only
    aa_embed = aa_embed[:,:128]

    torch.save(aa_embed, os.path.join(basedir, 'esm_embed.pt'))

    return aa_embed


def get_ss_graph_each_sample(args, sample, base_dir=None, cutoff=20):
    """

    :param args:
    :param sample:
    :param base_dir: path of saving dir. No data saving by default
    :param cutoff:
    :return:
    """

    pos = torch.tensor(sample['CA']).float()
    ss_seq = sample['ss_seq']  # ss_seq string
    ss_len = len(ss_seq)
    if args.split_with_ss8:
        ss_idx = torch.tensor([ss8_vocab[ss] for ss in ss_seq], dtype=torch.long)
    else:
        ss_idx = torch.tensor([ss3_vocab[ss] for ss in ss_seq], dtype=torch.long)

    # # TODO
    embedding = sample['embedding']
    max_len_per_ss = args.max_len_per_ss
    input_mask_list = []
    for j in range(ss_len):
        assert embedding[j].shape[0] <= max_len_per_ss, f"get aa length in one second structure longer than threshold: {max_len_per_ss}"
        input_mask_list.append(torch.cat([torch.ones(1, embedding[j].shape[0]), torch.zeros(1, max_len_per_ss - embedding[j].shape[0])], dim=-1))
    input_mask = torch.cat(input_mask_list, dim=0)
    # embedding = torch.cat([torch.unsqueeze(embedding[j].resize_(max_len_per_ss, embedding[j].shape[-1]), dim=0) for j in range(ss_len)], dim=0).cpu()
    # print(embedding.shape)

    raw_embedding = sample['embedding']
    embedding = torch.cat([torch.mean(raw_embedding[ii], dim=0, keepdim=True) for ii in range(ss_len)], dim=0).cpu()
    if args.split_with_ss8:
        struc_2nds_res_alphabet = struc_2nds_res_alphabet8
    else:
        struc_2nds_res_alphabet = struc_2nds_res_alphabet3

    rec_graph = Data(
        # x=embedding,  # [ss_len, 1280]
        b_type=F.one_hot(ss_idx, num_classes=len(struc_2nds_res_alphabet)),  # [ss_len,3] or [ss_len,8] one-hot
        b_pos=pos,
        raw_embedding=raw_embedding,
        input_mask=input_mask
    )

    #### compute 5-nn graph
    no_nn = 1
    distances_ = torch.cdist(pos, pos, p=2)  # [B, B]

    print(pos.norm())

    distances = distances_ <= cutoff
    distances = distances + 0
    distances = distances * distances_
    distances1 = np.triu(distances, k=1)
    distances2 = np.tril(distances, k=-1)
    src_list = []
    dst_list = []
    edges = []
    for i in range(1, pos.shape[0]):
        min_dist = {}
        edges.append((i - 1, i))
        for j in range(i + 1, pos.shape[0]):
            if distances1[i - 1][j] <= 1e-5:  # 不找0, 0都是cutoff被排除的
                continue
            if len(min_dist.keys()) < no_nn:
                min_dist[j] = distances1[i - 1][j]
                continue
            if max(min_dist.values()) > distances1[i - 1][j]:
                min_dist.pop(list(min_dist.keys())[np.argmax(list(min_dist.values()))])
                min_dist[j] = distances1[i - 1][j]
        for k in list(min_dist.keys()):
            edges.append((i - 1, k))
    for i in range(pos.shape[0] - 2, -1, -1):
        min_dist = {}
        if (i, i + 1) not in edges and (i + 1, i) not in edges:
            edges.append((i + 1, i))
        for j in range(i - 1, -1, -1):
            if distances1[i + 1][j] <= 1e-5:  # 不找0, 0都是cutoff被排除的
                continue
            if len(min_dist.keys()) < no_nn:
                min_dist[j] = distances2[i + 1][j]
                continue
            if max(min_dist.values()) > distances2[i + 1][j]:
                min_dist.pop(list(min_dist.keys())[np.argmax(list(min_dist.values()))])
                min_dist[j] = distances2[i + 1][j]
        for k in list(min_dist.keys()):
            if (i + 1, k) not in edges and (k, i + 1) not in edges:
                edges.append((i + 1, k))
    # print(edges)
    for e in edges:
        src_list.append(e[0])
        dst_list.append(e[1])
    edge_indexs = torch.tensor([src_list, dst_list], dtype=torch.long)
    dist_attr = []
    for i in range(edge_indexs.shape[-1]):
        dist_attr.append(1 / distances_[edge_indexs[0][i], edge_indexs[1][i]])

    dist_attr = torch.tensor(dist_attr).unsqueeze(-1)

    rec_graph.b_edge_index, rec_graph.b_edge_attr = edge_indexs, dist_attr

    if base_dir is not None:
        graph_save_dir = os.path.join(base_dir, 'graph')
        if not os.path.exists(graph_save_dir):
            os.mkdir(graph_save_dir)
        torch.save(rec_graph, os.path.join(graph_save_dir, sample['pdb'] + '.pt'))

    return rec_graph



def process_graph(basedir, safe_domi=1e-10):
    graph_dir = os.path.join(basedir, 'graph')
    save_process_dir = os.path.join(basedir, 'process')
    if not os.path.exists(save_process_dir):
        os.mkdir(save_process_dir) 

    filenames = os.listdir(graph_dir)
    random.shuffle(filenames)
    # calculating statistics feature for all data takes too much time, here we just sample a few
    n = len(filenames[:100])
    # all_idx = list(range(n))
    # random.shuffle(all_idx)

    # train_num = int(n * train_ratio)

    # init mean pos
    # TODO: there might be "NaN" in data ?
    graph = torch.load(os.path.join(graph_dir, filenames[0]))
    pos = graph.b_pos
    pos_mean = torch.zeros(pos.shape[1])
    pos_std = torch.zeros(pos.shape[1])
    print('processing graph data')
    for i in tqdm(range(n)):
        file = filenames[i]
        graph = torch.load(os.path.join(graph_dir, file))
        pos = graph.b_pos
        pos_mean += pos.mean(dim=0)
        pos_std += pos.std(dim=0)

    pos_mean = pos_mean.div_(n)
    pos_std = pos_std.div_(n)
    statistics = {'pos_mean': pos_mean, 'pos_std': pos_std}
    print(statistics)
    torch.save(statistics, os.path.join(basedir, 'graph_statistics'))


    print('saving...')
    # processed_files = os.listdir(save_process_dir)

    for file in tqdm(filenames):
        # if file in processed_files:
        #     continue
        graph = torch.load(os.path.join(graph_dir, file))
        graph.b_pos = graph.b_pos - graph.b_pos.mean(dim=-2, keepdim=False)
        graph.b_pos = graph.b_pos.div_(pos_std + safe_domi)

        torch.save(graph, os.path.join(save_process_dir, file))







if __name__ == "__main__":
    # working dir: protdiff-2nd/
    # run with python -m src.data.data_proc
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='CATH', choices=['CATH', 'test'],
                        help='which dataset used for training, CATH or TS')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='which dataset used for training, CATH or TS')
    parser.add_argument('--struc_2nds_res_path', type=str, default='ss_raw',
                        help='filepath to secondary structure')

    args = parser.parse_args()
    config = vars(args)

    basedir = os.path.join(config['data_dir'], config['dataset'])
    # ss_dir = os.path.join(basedir, config['struc_2nds_res_path'])
    #
    # CATH_test_inmem = Cath_imem(basedir, struc_2nds_res_path=ss_dir)

    # storing encoder embedding
    # get_store_esm_embeddings(basedir)

    get_ss_graph(basedir)
    process_graph(basedir)

