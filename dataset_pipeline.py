import json
import random
import os
import argparse
import shutil

import torch
from tqdm import tqdm
from Bio import PDB
from Bio.SeqUtils import seq1
from transformers import EsmModel, EsmConfig, AutoTokenizer
from src.data.extract_ss_embedding import generate_embedding, generate_ss2aa, dssp_process
from src.data.data_proc import process_graph, get_ss_graph_each_sample
from src.args import create_parser
from src.data.CATH import CATH_Domain_List_Process
# import matplotlib.pyplot as plt
import numpy as np
import re

def dataset_pipeline(args):
    dataset_dir = os.path.join(args.data_root, args.dataset)
    pdb_dir = os.path.join(dataset_dir, 'pdb')
    tokenizer = AutoTokenizer.from_pretrained(args.ESM2_dir)
    model = EsmModel.from_pretrained(args.ESM2_dir)
    model.cuda()
    model.eval()
    pdb_files = sorted(os.listdir(pdb_dir))
    if len(pdb_files) > args.max_process_pdb_num:
        pdb_files = pdb_files[:args.max_process_pdb_num]
    aa_feature_out_dir = os.path.join(dataset_dir, 'aa_feature')
    graph_out_dir = os.path.join(dataset_dir, 'graph')
    os.makedirs(aa_feature_out_dir, exist_ok=True)
    os.makedirs(graph_out_dir, exist_ok=True)
    aa_feature_list = os.listdir(aa_feature_out_dir)
    graph_list = os.listdir(graph_out_dir)


    for p in tqdm(pdb_files):
        ret = re.match('\w+\.pdb', p)
        if not ret:
            os.rename(os.path.join(pdb_dir, p), os.path.join(pdb_dir, p+'.pdb'))
            p = p+'.pdb'
        try:
            file_name = p.split('.')[0] + '.pt'
            aa_feature_out_file = os.path.join(aa_feature_out_dir, file_name)
            graph_out_file = os.path.join(graph_out_dir, file_name)
            if (file_name in aa_feature_list) and (file_name in graph_list):
                continue
            dssp_data = dssp_process(os.path.join(pdb_dir, p))
            feature = generate_embedding(args, os.path.join(pdb_dir, p), model, tokenizer, dssp_data)
            feature_decoder = generate_ss2aa(os.path.join(pdb_dir, p), dssp_data)
            get_ss_graph_each_sample(args, feature, dataset_dir)
            torch.save(feature_decoder, aa_feature_out_file)
        except Exception as e:
            print('error while processing: ', p)
            print('Exception: ', e)
    process_graph(dataset_dir)

def pdb_pipeline(args, pdb_file):
    """
    process a single pdbfile
    :param args:
    :return:
    """
    tokenizer = AutoTokenizer.from_pretrained(args.ESM2_dir)
    model = EsmModel.from_pretrained(args.ESM2_dir)
    model.cuda()
    model.eval()

    dssp_data = dssp_process(pdb_file)
    feature = generate_embedding(args, pdb_file, model, tokenizer, dssp_data)
    # feature_decoder = generate_ss2aa(pdb_file, dssp_data)
    graph = get_ss_graph_each_sample(args, feature)
    return graph


# draw histogram of second structure length
def len_per_ss_count(args):
    dataset_dir = os.path.join(args.data_root, args.dataset)
    pdb_dir = os.path.join(dataset_dir, 'pdb')
    pdb_files = os.listdir(pdb_dir)
    random.shuffle(pdb_files)
    if len(pdb_files) > args.max_process_pdb_num:
        pdb_files = pdb_files[:args.max_process_pdb_num]
    out_dir = os.path.join(dataset_dir, 'aa_feature')
    os.makedirs(out_dir, exist_ok=True)
    random.shuffle(pdb_files)
    len_per_ss3_list = []
    len_per_ss8_list = []
    len_per_ss3_code = {'C': [], 'E': [], 'H': []}
    max_len_per_ss3_list = []
    max_len_per_ss8_list = []
    for p in tqdm(pdb_files):
        try:
            len_per_ss3_list_ = []
            len_per_ss8_list_ = []
            feature_decoder = generate_ss2aa(os.path.join(pdb_dir, p))
            for key in feature_decoder['ss3_to_aa'].keys():
                ss_str = feature_decoder['ss3_to_aa'][key]
                len_per_ss3_list_.append(len(ss_str))
                len_per_ss3_code[feature_decoder['ss3_seq'][key]].append(len(ss_str))
            for key in feature_decoder['ss8_to_aa'].keys():
                ss_str = feature_decoder['ss8_to_aa'][key]
                len_per_ss8_list_.append(len(ss_str))
            len_per_ss8_list.extend(len_per_ss8_list_)
            len_per_ss3_list.extend(len_per_ss3_list_)
            max_len_per_ss8_list.append(max(len_per_ss8_list_))
            max_len_per_ss3_list.append(max(len_per_ss3_list_))
        except Exception as e:
            print('error while processing: ', p)
            print('Exception: ', e)
    len_per_ss3 = np.array(len_per_ss3_list)
    len_per_ss8 = np.array(len_per_ss8_list)
    max_len_per_ss3 = np.array(max_len_per_ss3_list)
    max_len_per_ss8 = np.array(max_len_per_ss8_list)
    # plt.hist(len_per_ss3, log=True, bins=200, color='blue', alpha=0.5)
    # plt.title('Histogram of Second Structure length (3 types)')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.savefig('results/figure/ss3_len_histogram.png')
    # plt.clf()
    # plt.hist(len_per_ss8, log=True, bins=200, color='red', alpha=0.5)
    # plt.title('Histogram of Second Structure length (8 types)')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.savefig('results/figure/ss8_len_histogram.png')
    # plt.clf()
    # plt.hist(max_len_per_ss3, log=True, bins=200, color='blue', alpha=0.5)
    # plt.title('Histogram of Max Second Structure length (3 types)')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.savefig('results/figure/max_ss3_len_histogram.png')
    # plt.clf()
    # plt.hist(max_len_per_ss8, log=True, bins=200, color='red', alpha=0.5)
    # plt.title('Histogram of Max Second Structure length (8 types)')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.savefig('results/figure/max_ss8_len_histogram.png')
    # plt.clf()
    # for key in len_per_ss3_code.keys():
    #     plt.hist(len_per_ss3_code[key], log=True, bins=200, color='blue', alpha=0.5)
    #     plt.title(f'Histogram of Second Structure {key} length (3 types)')
    #     plt.xlabel('Value')
    #     plt.ylabel('Frequency')
    #     plt.savefig(f'results/figure/ss3_{key}_len_histogram.png')
    #     plt.clf()


    torch.save(len_per_ss3_code, "results/weight/len_per_ss3_code.pt")
    torch.save(torch.from_numpy(len_per_ss3), "results/weight/ss3.pt")
    torch.save(torch.from_numpy(len_per_ss8), "results/weight/ss8.pt")
    torch.save(max_len_per_ss8_list, "results/weight/max_len_per_ss8_list.pt")
    torch.save(max_len_per_ss3_list, "results/weight/max_len_per_ss3_list.pt")

def remove_invalid_pdb_files(args):
    dataset_dir = os.path.join(args.data_root, args.dataset)
    pdb_dir = os.path.join(dataset_dir, 'pdb')
    pdb_files = os.listdir(pdb_dir)
    for file in tqdm(pdb_files):
        ret = re.match('\w+\.pdb', file)
        if not ret:
            os.remove(os.path.join(pdb_dir, file))

def split_CATH_with_label():
    """
    before split data with CATH number(label), make sure:
    1. process the raw dompdb files downloaded from CATH website using dataset_pipeline()
    2. the processed data shall be stored in four dirs ['pdb', 'aa_feature', 'graph', 'process']
    3. get all the processed data into dir 'CATH43_S20_SPLIT_TRAIN_VAL' or 'CATH43_S40_SPLIT_TRAIN_VAL'

    :return:
    """
    args = create_parser()
    split_dataset = [['CATH43_S20_SPLIT_TRAIN_VAL', 'CATH43_S20_SPLIT_TEST'], ['CATH43_S40_SPLIT_TRAIN_VAL', 'CATH43_S40_SPLIT_TEST']]
    data_type = ['pdb', 'aa_feature', 'graph', 'process']

    # get label_dict for CATH data
    cath_domain_list_path = os.path.join(args.data_root, 'cath-domain-list.json')
    label_dict = CATH_Domain_List_Process(cath_domain_list_path)
    cath_path = os.path.join(args.data_root, split_dataset[1][0])
    cath_pdb_path = os.path.join(cath_path, 'pdb')


    # categorize CATH data based on CATH number(label)
    CATH_file_dict = dict()
    test_split = list()
    for file_name in os.listdir(cath_pdb_path):
        name = file_name.split('.')[0]
        CATH_key = ['C', 'A', 'T', 'H']
        CATH_num = '.'.join([label_dict[name][key] for key in CATH_key])
        if CATH_num not in CATH_file_dict.keys():
            CATH_file_dict[CATH_num] = list()
        CATH_file_dict[CATH_num].append(name)

    # split train_val & test set
    for CATH_num in CATH_file_dict.keys():
        length = len(CATH_file_dict[CATH_num])
        print(CATH_num, 'length:', length)
        random.shuffle(CATH_file_dict[CATH_num])
        test_split.extend(CATH_file_dict[CATH_num][:int(length*args.test_ratio)])
    print('test split length:', len(test_split))
    print('total length:', len(os.listdir(cath_pdb_path)))

    # move
    target_path = os.path.join(args.data_root, split_dataset[1][1])
    for data_type_dir in data_type:
        src_dir_path = os.path.join(cath_path, data_type_dir)
        dst_dir_path = os.path.join(target_path, data_type_dir)
        os.makedirs(dst_dir_path, exist_ok=True)
        for name in test_split:
            if data_type_dir == 'pdb':
                src_path = os.path.join(src_dir_path, name+'.pdb')
            else:
                src_path = os.path.join(src_dir_path, name+'.pt')
            shutil.move(src_path, dst_dir_path)

        










if __name__ == '__main__':
    # run with python dataset_pipeline.py --dataset sample
    # working dir protdiff-2nd/

    args = create_parser()
    dataset_pipeline(args)




