import json
import functools
import torch
import os.path
from dataclasses import dataclass
from src.args import create_parser
import subprocess
import shutil
from src.data.extract_ss_embedding import generate_ss2aa
from src.data.esmfold import call_esmfold
from Bio.Align import PairwiseAligner
import numpy as np
from tqdm import tqdm
import random
import esm
import re
import csv
import pickle
import networkx as nx
from transformers import EsmModel, EsmConfig, AutoTokenizer, EsmForMaskedLM, EsmTokenizer
# from torch_geometric.data import Data
from Bio import SeqIO, PDB, SeqRecord, Seq
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.DSSP import DSSP
import Bio
from Bio.SeqUtils import seq1
from math import pi, log10

import pandas as pd
import matplotlib.pyplot as plt
from src.data.foldseek_web import FoldseekClient
# conda install -c conda-forge -c bioconda mmseqs2
# conda install -c conda-forge -c bioconda foldseek
# conda install -c schrodinger tmalign

import dataset_pipeline


@dataclass
class ProtDiffConfig:
    name: str 
    diff_ckpt: str
    decoder_ckpt: str
    encoder_type: str
    n_layers: str
    hdim: str
    embedding_dim: str
    diff_dp: str
    noise_schedule: str
    type: str = "ss-condition"

@dataclass
class RFdiffusionConfig:
    name: str = 'RFdiffusion'
    type: str = "ss-condition"

@dataclass
class VanillaDecoderConfig:
    name: str
    decoder_ckpt: str
    encoder_type: str
    type: str = 'encoder_embedding'  # random_embedding/encoder_embedding


# ProtDiffS20Config = ProtDiffConfig(
#     name='protdiff_2nd_S20',
#     diff_ckpt='results/diffusion/weight/20240312/diffusion_CATH43S20.pt',
#     decoder_ckpt='./results/decoder/ckpt/20240227/decoder_ESM2_AFDB_AttentionPooling.pt',
#     encoder_type='AttentionPooling'
# )


@dataclass
class ESMIFConfig:
    name: str
    model: object
    type: str = "IF"


@dataclass
class ESM2Config:
    name: str
    model: object
    tokenizer: object
    mask_ratio: float = 0.2
    type: str = 'PLM'

@dataclass
class ESM3Config:
    name: str = 'esm_3'
    type: str = "variable_length" # or "ss-condition" to use fixed length design

@dataclass
class prostT5Config:
    name: str = 'prostT5'
    type: str = 'foldseek_PLM'


@dataclass
class ProteinMPNNConfig:
    name: str = 'ProteinMPNN'
    ckpt: str = os.path.join('baselines', 'ProteinMPNN', 'vanilla_model_weights')
    type: str = "IF"

def extract_sequence_from_pdb(pdb_file):
    """
    Extracts amino acid sequences from a PDB file.

    Args:
    - pdb_file (str): Path to the PDB file.

    Prints:
    - Amino acid sequences for each chain in the PDB file.
    """
    # 创建PDB解析器对象
    parser = PDB.PDBParser()

    # 解析PDB文件
    structure = parser.get_structure("MyStructure", pdb_file)

    # 遍历模型、链和残基，提取氨基酸序列
    model = structure[0]
    ppb = PDB.PPBuilder()
    pp = ppb.build_peptides(model)

    sequence = pp[0].get_sequence()
    return str(sequence)


@dataclass
class BenchmarkConfig:
    train_set: str = 'CATH43_S40_SPLIT_TRAIN_VAL'   # used for novelty measurement
    test_set: str = 'CATH43_S40_SPLIT_TEST'
    test_num: str = '50'
    sample_num: str = '200'


class ColabFoldParser:
    def __init__(self, colabfold_out_dir):
        # naming rules for seqs folded are as follows:
        # 1. generated seqs(i is a number): template_name + '_' + str(i)
        # 2. original seqs: template_name + '_original'
        
        self.colabfold_output_dir = colabfold_out_dir
        self.files = self.get_files()
        self.folding_names, self.template_names = self.get_names()

    
    def get_files(self):
        list_dir = os.listdir(self.colabfold_output_dir)
        files = list()
        for file in list_dir:
            if os.path.isfile(os.path.join(self.colabfold_output_dir, file)):
                files.append(file)
        return files
    
    def get_names(self):
        folding_names = list()
        for file_name in self.files:
            ret = re.match('(\w+)_plddt', file_name)
            if ret:
                folding_names.append(ret.group(1))

        # find template names
        template_names = list()
        for name in folding_names:
            ret = re.match('(\w+)_original', name)
            if ret:
                template_names.append(ret.group(1))

        return folding_names, template_names
    

    def get_avg_plddt(self, folding_name: str):
        for file in self.files:
            ret = re.match(folding_name + '_scores_rank_001\w+\.json', file)
            if ret:
                plddt_json = ret.group(0)
                break
        else:
            raise Exception("invalid folding name")
        with open(os.path.join(self.colabfold_output_dir, plddt_json), 'r') as f:
            scores_dict = json.load(f)
        plddt_score = np.array(scores_dict['plddt'])
        return plddt_score.mean().item()
                
    def get_path(self, folding_name: str):
        # find pdb path for each folding
        for file in self.files:
            ret = re.match(folding_name + '_relaxed_rank_001\w+\.pdb', file)
            if ret:
                pdb_name = ret.group(0)
                break
        else:
            for file in self.files:
                ret = re.match(folding_name + '_unrelaxed_rank_001\w+\.pdb', file)
                if ret:
                    pdb_name = ret.group(0)
                    break
            else:
                raise Exception("cannot find corresponding pdb file")

        pdb = os.path.join(self.colabfold_output_dir, pdb_name)
        plddt_plot = os.path.join(self.colabfold_output_dir, folding_name + '_plddt.png')
        pae_plot = os.path.join(self.colabfold_output_dir, folding_name + '_pae.png')
        return pdb, plddt_plot, pae_plot


def second_structure_pdb(in_pdb, out_dir):
    """
    modification to the input pdb files for Inverse Folding Models.
    change all atom positions to the positions of the secondary structure it belongs to
    :param in_pdb: pdb file path
    :param out_dir:
    :return:
    """
    print("processing pdb file for Inverse Folding Models...")
    p = PDBParser(PERMISSIVE=True)
    template_name = in_pdb.split('/')[-1].split('.')[0]
    path = in_pdb.split('/')
    path[-2] = 'aa_feature'
    path[-1] = path[-1].replace(".pdb", ".pt")
    aa_feature_path = os.path.join(*path)
    aa_feature = torch.load(aa_feature_path)
    ss3_to_aa = aa_feature['ss3_to_aa']
    print(ss3_to_aa)
    structure = p.get_structure(template_name, in_pdb)

    # extract amino acid sequence
    seq = []
    for residue in structure.get_residues():
        if residue.get_id()[0] == " ":
            seq.append(residue.get_resname())
    one_letter_seq = "".join([seq1(aa) for aa in seq])

    coords = {"N": [], "CA": [], "C": [], "O": []}
    ss_ids = list(ss3_to_aa.keys())
    ss_ids.sort()
    for residue in structure.get_residues():
        if residue.get_id()[0] == " ":
            for atom_name in coords.keys():
                if residue.has_id(atom_name):
                    atom = residue[atom_name]
                    coords[atom_name].append(atom.get_coord())
                else:
                    coords[atom_name].append('NaN')
    def remove_nan(x: list):
        return [t for t in x if not type(t) is str]

    offset = 0
    multi_ss_coords = list()
    for ss_id in ss_ids:
        ss_coords = {"N": 'NaN', "CA": 'NaN', "C": 'NaN', "O": 'NaN'}
        for atom_name in coords.keys():
            ss_coord = remove_nan(coords[atom_name][offset:offset + len(ss3_to_aa[ss_id])])
            if len(ss_coord) > 0:
                ss_coords[atom_name] = np.mean(np.array(ss_coord), axis=0, keepdims=False)
        multi_ss_coords.append(ss_coords)

    ss_id_ptr=0
    aa_in_per_ss_ptr=0
    for residue_ptr, residue in enumerate(structure.get_residues()):
        if residue.get_id()[0] == " ":
            aa_in_per_ss_ptr += 1
            if aa_in_per_ss_ptr > len(ss3_to_aa[ss_id_ptr]):
                aa_in_per_ss_ptr = 1
                ss_id_ptr += 1
            for atom_name in coords.keys():
                if residue.has_id(atom_name):
                    atom = residue[atom_name]
                    if not type(multi_ss_coords[ss_id_ptr][atom_name]) is str:
                        atom.set_coord(multi_ss_coords[ss_id_ptr][atom_name])

    io = PDBIO()
    io.set_structure(structure)
    os.makedirs(out_dir, exist_ok=True)
    io.save(os.path.join(out_dir, in_pdb.split('/')[-1]))


class PathConfig:
    def __init__(self, benchmark_config):
        self.benchmark_config = benchmark_config

        # path settings
        self.base_dir = os.path.join('experiments', self.benchmark_config.test_set)
        self.log_dir = os.path.join(self.base_dir, 'progress')
        # design_pipeline
        self.design_dir = os.path.join(self.base_dir, 'design_pipeline')
        self.seq_dir = os.path.join(self.design_dir, 'seqs')
        self.esmfold_sructure_dir = os.path.join(self.design_dir, 'esmfold_structures')
        self.colabfold_sructure_dir = os.path.join(self.design_dir, 'colabfold_sructures')
        self.plddt_filter_structure_dir = os.path.join(self.design_dir, 'plddt_filter_structures')
        self.design_filter_structure_dir = os.path.join(self.design_dir, 'design_filter_structures')
        self.ss_dir = os.path.join(self.design_dir, 'second_structures')
        self.tmp_base = os.path.join(self.design_dir, 'tmp')    # this dir saves itermediate results and will be cleaned
        self.intermediate_base = os.path.join(self.design_dir, 'intermediate')   # this dir saves important itermediate results and will not be cleaned
        # benchmark_pipeline
        self.metrics_dir = os.path.join(self.base_dir, 'metrics')
        self.esmfold_metrics_dir = os.path.join(self.metrics_dir, 'esmfold_metrics')
        self.colabfold_metrics_dir = os.path.join(self.metrics_dir, 'colabfold_metrics')
        self.seq_metrics_dir = os.path.join(self.metrics_dir, 'seq_metrics')
        self.foldseek_base = os.path.join(self.metrics_dir, 'foldseek_tmp')

        os.makedirs('experiments', exist_ok=True)
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.design_dir, exist_ok=True)
        os.makedirs(self.seq_dir, exist_ok=True)
        os.makedirs(self.esmfold_sructure_dir, exist_ok=True)
        os.makedirs(self.colabfold_sructure_dir, exist_ok=True)
        os.makedirs(self.plddt_filter_structure_dir, exist_ok=True)
        os.makedirs(self.design_filter_structure_dir, exist_ok=True)
        os.makedirs(self.ss_dir, exist_ok=True)
        os.makedirs(self.tmp_base, exist_ok=True)
        os.makedirs(self.design_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.esmfold_metrics_dir, exist_ok=True)
        os.makedirs(self.colabfold_metrics_dir, exist_ok=True)
        os.makedirs(self.seq_metrics_dir, exist_ok=True)
        os.makedirs(self.foldseek_base, exist_ok=True)


        # self.clean_dirs([self.tmp_dir, self.foldseek_dir])

        print('Benchmark config:', self.benchmark_config.__dict__)

    @staticmethod
    def clean_dirs(path):
        if type(path) is str:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
        elif type(path) is list:
            for path_ in path:
                PathConfig.clean_dirs(path_)
        else:
            raise Exception(f"Wrong Path: {path}")


class DesignPipeline(PathConfig):
    def __init__(self, model_config, restart=False, benchmark_config=BenchmarkConfig(), fold_continue=False):
        super().__init__(benchmark_config)
        self.model_name = model_config.name
        self.model_config = model_config
        self.restart = restart
        self.fold_continue = fold_continue
        # rename special input model
        if self.model_config.type == "IF" or self.model_config.type == "PLM" or self.model_config.type == 'foldseek_PLM' or self.model_config.type == 'variable_length':
            # extra process for IF/PLM model input
            self.model_name += f"_{self.model_config.type}"
            self.model_config.name = self.model_name
        print('Model config:', model_config.__dict__)

        self.intermediate_dir = os.path.join(self.intermediate_base, self.model_name)
        self.tmp_dir = os.path.join(self.tmp_base, self.model_name)
        self.foldseek_dir = os.path.join(self.foldseek_base, self.model_name)
        self.clean_dirs([self.tmp_dir, self.foldseek_dir])


        self.load_log()
        self.data_list = self.select_sample_set()

    def progress_control(pipeline_process):
        """
        (This is a decorator)
        control progress in pipeline, skip process which has been done before (set to 'done' in progress.json)
        """
        @functools.wraps(pipeline_process)
        def wrapper(self, *args, **kwargs):
            result = None
            if pipeline_process.__name__ in self.log.keys() and self.log[pipeline_process.__name__] == 'done':
                print(f'skipping process {pipeline_process.__name__}')
            else:
                # try:
                result = pipeline_process(self, *args, **kwargs)
                print(f'{pipeline_process.__name__}: done')
                self.log[pipeline_process.__name__] = 'done'
                self.save_log()
                # except Exception as e:
                #     print(f'error occurred running {pipeline_process.__name__}: {e}')
            return result
        return wrapper

    def rerun(self, func, *args, **kwargs):
        """igonre log, and rerun the given function"""
        self.log[func.__name__] = "rerun"
        print("rerun", self.log)
        return func(*args, **kwargs)

    def run(self, length_filter=10000):
        self.model_init()
        self.seq_generation()
        self.esmfold(length_filter=length_filter)
        self.dssp_annotation()

    @ progress_control
    def model_init(self):
        if re.match('prostT5', self.model_name):
            # save sample_3di.fasta to test_set dir as prostT5 input
            test_set_dir = os.path.join(args.data_root, self.benchmark_config.test_set)
            out_fasta = os.path.join(test_set_dir, 'sample_3di.fasta')
            selected_pdb_dir = os.path.join(self.tmp_dir, 'selected_pdb')
            os.makedirs(selected_pdb_dir, exist_ok=True)
            print("processing dataset with length", len(self.data_list))
            for pdb_file in self.data_list:
                pdb_path = os.path.join(test_set_dir, 'pdb', pdb_file)
                shutil.copy(pdb_path, selected_pdb_dir)
            # generate 3Di for prostT5 input
            pdb_to_3di_shell_path = os.path.join('script', 'shell', 'pdb_to_3di.sh')
            command = f"bash {pdb_to_3di_shell_path} {selected_pdb_dir} {self.tmp_dir} {out_fasta}"
            result = subprocess.run(command, shell=True, encoding="utf-8", check=True)
            if result.stderr:
                print("Error in Foldseek:", result.stderr)

            if self.model_config.type == 'foldseek_PLM':
                from transformers import T5Tokenizer
                tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5')
                mask_token = tokenizer.unk_token
                print('mask token', mask_token)
                print("mask foldseek tokens")
                # mask foldseek tokens
                foldseek_dict = dict()
                with open(out_fasta, 'r') as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        if re.match('^>', line):
                            key = line[1:]
                            foldseek_dict[key] = ''
                        else:
                            foldseek_dict[key] += line
                # print('original foldseek seq:', foldseek_dict)
                # mask foldseek seq
                for key, seq in foldseek_dict.items():
                    aa_feature_file_path = os.path.join(args.data_root, self.benchmark_config.test_set, 'aa_feature', key.replace(".pdb", ".pt"))
                    aa_feature = torch.load(aa_feature_file_path)
                    foldseek_dict[key] = self.plm_mask(seq, aa_feature, mask_token=mask_token)
                with open(out_fasta, 'w') as f:
                    content = list()
                    for key, seq in foldseek_dict.items():
                        content.append(f'>{key}')
                        content.append(seq)
                    f.write("\n".join(content))

        self.clean_dirs(self.tmp_dir)

    def select_sample_set(self):
        """
        select sample set with length <= test_num from test_set
        (manually set random seed)
        :return:
        """
        data_list = os.listdir(os.path.join(args.data_root, self.benchmark_config.test_set, 'process'))
        random.seed(args.random_seed)
        random.shuffle(data_list)
        if len(data_list) >= int(self.benchmark_config.test_num):
            data_list = data_list[:int(self.benchmark_config.test_num)]
        data_list = [data.split('.')[0]+'.pdb' for data in data_list]
        print(">>> sample set length: ", len(data_list))
        return data_list

    @staticmethod
    def plm_mask(seq, aa_feature, mask_token='<mask>'):
        """
        for every secondary structure, leave one AA remain unmasked and mask others
        :param seq:
        :return:
        """
        ori_seq = list(seq)
        masked_seq = [mask_token] * len(seq)
        offset = 0
        for key, aa_str in aa_feature['ss3_to_aa'].items():
            ss_len = len(aa_str)
            unmask_indice = random.sample(range(ss_len), 1)[0]
            masked_seq[offset + unmask_indice] = ori_seq[offset + unmask_indice]
            offset += ss_len
        return "".join(masked_seq)

    @staticmethod
    def get_ss_seq(aa_feature, ss3=True, variable_length=True):
        """
        get sequence of secondary structures (not secondary structure elements)
        """
        ss_seq = ""
        ss_type_num = 3 if ss3 else 8
        for ss_index, ss_type in enumerate(aa_feature[f"ss{ss_type_num}_seq"]):
            wildtype_length = len(aa_feature[f"ss{ss_type_num}_to_aa"][ss_index])
            if variable_length: # this setting is based on the statistics of ss3
                if ss_type == 'H':  # length of helix should be over 3
                    length = random.randint(3, max(3, 2 * wildtype_length))
                else:   # "E" OR "C"
                    length = random.randint(1, max(1, 2 * wildtype_length))
            else:
                length = wildtype_length
            ss_seq += ss_type*length
        return ss_seq


    @progress_control
    def seq_generation(self):
        """
        step1: generate sequences with the given model
        :return:
        """
        print("generating sequences...")
        out_dir = os.path.join(self.seq_dir, self.model_name)
        self.clean_dirs(out_dir)
        if re.match('protdiff_2nd', self.model_name):
            command = ['python train_encoder_decoder.py']
            command.extend(['--sample'])
            command.extend(['--sample_out_split'])
            command.extend(['--sample_out_dir', out_dir])
            command.extend(['--encoder_type', self.model_config.encoder_type])
            command.extend(['--diff_ckpt', self.model_config.diff_ckpt])
            command.extend(['--decoder_ckpt', self.model_config.decoder_ckpt])
            command.extend(['--dataset', self.benchmark_config.test_set])
            command.extend(['--max_data_num', self.benchmark_config.test_num])
            command.extend(['--sample_nums', self.benchmark_config.sample_num])

            command.extend(['--n_layers', self.model_config.n_layers])
            command.extend(['--hdim', self.model_config.hdim])
            command.extend(['--diff_dp', self.model_config.diff_dp])
            command.extend(['--embedding_dim', self.model_config.embedding_dim])
            command.extend(['--noise_schedule', self.model_config.noise_schedule])

            # command.extend(['--min_len_filter', '0.5'])
            # command.extend(['--max_len_filter', '1.5'])

            # command = 'python train_encoder_decoder.py --sample --sample_nums 10 --sample_out_split --dataset sample --diff_ckpt results/diffusion/weight/20240312/diffusion_CATH43S20.pt --decoder_ckpt ./results/decoder/ckpt/20240227/decoder_ESM2_AFDB_AttentionPooling.pt --sample_out_dir benchmark/seqs/protdiff-2nd --encoder_type AttentionPooling'
            command = ' '.join(command)
            print('running command:', command)
            ret = subprocess.run(command, shell=True, encoding="utf-8", check=True)
        elif re.match('esm_if', self.model_name):
            import esm.inverse_folding

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for pdb_file in tqdm(self.data_list, desc='sample'):
                pdb_path = os.path.join(args.data_root, self.benchmark_config.test_set, 'pdb', pdb_file)
                if self.model_config.type == 'IF':
                    new_pdb_dir = os.path.join(args.data_root, self.benchmark_config.test_set, 'pdb__')
                    second_structure_pdb(pdb_path, new_pdb_dir)
                    pdb_path = os.path.join(new_pdb_dir, pdb_file)
                seq_name = pdb_file.split('.')[-2]
                out_path = os.path.join(out_dir, seq_name + '_original.fasta')
                coords, original_seq = esm.inverse_folding.util.load_coords(pdb_path, seq_name[-3])
                with open(out_path, 'w') as f:
                    f.write('>' + seq_name + '_original' + '\n' + original_seq)
                for num in tqdm(range(int(self.benchmark_config.sample_num))):
                    out_path = os.path.join(out_dir, seq_name + '_' + str(num) + '.fasta')
                    sampled_seq = self.model_config.model.sample(coords, temperature=1, device=device)
                    with open(out_path, 'w') as f:
                        f.write('>' + seq_name + '_' + str(num) + '\n' + sampled_seq)
        elif re.match('prostT5', self.model_name):
            import esm.inverse_folding

            for pdb_file in self.data_list:
                pdb_file_path = os.path.join(args.data_root, self.benchmark_config.test_set, 'pdb', pdb_file)
                seq_name = pdb_file.split('.')[-2]
                out_path = os.path.join(out_dir, seq_name + '_original.fasta')
                coords, original_seq = esm.inverse_folding.util.load_coords(pdb_file_path, seq_name[-3])
                with open(out_path, 'w') as f:
                    f.write('>' + seq_name + '_original' + '\n' + original_seq)

            chunk_size = 20
            chunk_num = int(int(self.benchmark_config.sample_num)/chunk_size)
            in_fasta = os.path.join(args.data_root, self.benchmark_config.test_set, 'sample_3di.fasta')
            prostT5_script = os.path.join('baselines', 'ProstT5', 'scripts', 'translate.py')
            command = ['python', prostT5_script]
            command.extend(['--input', in_fasta])
            command.extend(['--output', out_dir])
            command.extend(['--half', '1'])
            command.extend(['--is_3Di', '1'])
            command.extend(['--num_return_sequences', str(chunk_size)])
            command.extend(['--time_return_sequences', str(chunk_num)])

            command = ' '.join(command)
            result = subprocess.run(command, shell=True, encoding="utf-8", check=True)
        elif re.match('ProteinMPNN', self.model_name):
            ProteinMPNN_script = os.path.join('baselines', 'ProteinMPNN', 'protein_mpnn_run.py')
            command = ['python', ProteinMPNN_script]
            command.extend(['--path_to_model_weights', self.model_config.ckpt])
            command.extend(['--num_seq_per_target', str(self.benchmark_config.sample_num)])
            command.extend(['--out_folder', self.tmp_dir.replace("\\", "/").split(":")[-1]])    # proteinMPNN script cannot correctly process windows path
            command.extend(['--pdb_path', 'path_to_pdb'])


            for pdb_file in tqdm(self.data_list, desc='sample'):
                pdb_path = os.path.join(args.data_root, self.benchmark_config.test_set, 'pdb', pdb_file)
                if self.model_config.type == 'IF':
                    new_pdb_dir = os.path.join(args.data_root, self.benchmark_config.test_set, 'pdb__')
                    second_structure_pdb(pdb_path, new_pdb_dir)
                    pdb_path = os.path.join(new_pdb_dir, pdb_file)

                command[-1] = pdb_path.replace("\\", "/").split(":")[-1]
                command_str = ' '.join(command)
                print('running command:', command_str)
                result = subprocess.run(command_str, shell=True, encoding="utf-8", check=True)

            # convert ProteinMPNN output fasta
            proteinMPNN_out_path = os.path.join(self.tmp_dir, 'seqs')
            proteinMPNN_out_files = os.listdir(proteinMPNN_out_path)
            for proteinMPNN_out_file in proteinMPNN_out_files:
                template_name = proteinMPNN_out_file.split('.')[0]
                file_path = os.path.join(proteinMPNN_out_path, proteinMPNN_out_file)
                with open(file_path, 'r') as f:
                    records = list(SeqIO.parse(f, "fasta"))
                records[0].id = template_name+'_original'
                records[0].description = ''
                out_file = records[0].id+'.fasta'
                out_path = os.path.join(out_dir, out_file)
                SeqIO.write(records[0], out_path, "fasta-2line")

                for i, record in enumerate(records[1:]):
                    record.id = template_name+'_'+str(i)
                    record.description = ''
                    out_file = record.id + '.fasta'
                    out_path = os.path.join(out_dir, out_file)
                    SeqIO.write(record, out_path, "fasta-2line")
        elif re.match('esm_2', self.model_name):
            import esm.inverse_folding

            def random_mask(seq, ratio=0.2):
                """
                Randomly mask characters in a string based on the given mask ratio.

                Args:
                    seq (str): Input string.
                    mask_ratio (float): Ratio of characters to mask, should be between 0 and 1.

                Returns:
                    str: String with characters randomly masked.
                """
                num_chars_to_mask = int(len(seq) * ratio)

                mask_indices = random.sample(range(len(seq)), num_chars_to_mask)

                masked_string = list(seq)
                for index in mask_indices:
                    masked_string[index] = "<mask>"

                return "".join(masked_string)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for pdb_file in tqdm(self.data_list, desc='sample'):
                pdb_file_path = os.path.join(args.data_root, self.benchmark_config.test_set, 'pdb', pdb_file)
                aa_feature_file_path = os.path.join(args.data_root, self.benchmark_config.test_set, 'aa_feature', pdb_file.replace(".pdb", ".pt"))
                seq_name = pdb_file.split('.')[-2]
                out_path = os.path.join(out_dir, seq_name + '_original.fasta')
                coords, original_seq = esm.inverse_folding.util.load_coords(pdb_file_path, seq_name[-3])
                aa_feature = torch.load(aa_feature_file_path)

                with open(out_path, 'w') as f:
                    f.write('>' + seq_name + '_original' + '\n' + original_seq)
                for num in tqdm(range(int(self.benchmark_config.sample_num))):
                    out_path = os.path.join(out_dir, seq_name + '_' + str(num) + '.fasta')
                    if self.model_config.type == "PLM":
                        seq = self.plm_mask(original_seq, aa_feature)
                    else:
                        seq = random_mask(original_seq, ratio=self.model_config.mask_ratio)
                    inputs = self.model_config.tokenizer(seq, return_tensors="pt")
                    input_ids = inputs["input_ids"].cuda()
                    with torch.no_grad():
                        outputs = self.model_config.model(input_ids)
                        logits = outputs['logits']
                        prediction = logits.argmax(dim=-1, keepdim=False)
                        sampled_seq = re.sub(r"\s+", "",
                                     self.model_config.tokenizer.decode(prediction[0], skip_special_tokens=True))
                    with open(out_path, 'w') as f:
                        f.write('>' + seq_name + '_' + str(num) + '\n' + sampled_seq)
        elif re.match('VanillaDecoder', self.model_name):
            command = ['python train_encoder_decoder.py']
            command.extend(['--sample'])
            command.extend(['--sample_out_split'])
            command.extend(['--sample_out_dir', out_dir])
            command.extend(['--encoder_type', self.model_config.encoder_type])
            command.extend(['--decoder_ckpt', self.model_config.decoder_ckpt])
            command.extend(['--dataset', self.benchmark_config.test_set])
            command.extend(['--max_data_num', self.benchmark_config.test_num])
            command.extend(['--sample_nums', self.benchmark_config.sample_num])

            command.extend(['--min_len_filter', '0.0'])
            command.extend(['--max_len_filter', '100.0'])

            if self.model_config.type == "random_embedding":
                command.extend(['--random_embedding', 'True'])
            elif self.model_config.type == "encoder_embedding":
                command.extend(['--random_embedding', 'False'])

            # command = 'python train_encoder_decoder.py --sample --sample_nums 10 --sample_out_split --dataset sample --diff_ckpt results/diffusion/weight/20240312/diffusion_CATH43S20.pt --decoder_ckpt ./results/decoder/ckpt/20240227/decoder_ESM2_AFDB_AttentionPooling.pt --sample_out_dir benchmark/seqs/protdiff-2nd --encoder_type AttentionPooling'
            command = ' '.join(command)
            print('running command:', command)
            ret = subprocess.run(command, shell=True, encoding="utf-8", check=True)
        elif re.match('RFdiffusion', self.model_name):
            # step 1: run ss_adj_script in RFdiffsuion repo to get target_ss.pt & target_adj.pt
            # step 2: run rfdiffusion_script to generate monomer backbone
            # step 3: generate sequences using ProteinMPNN

            # configurations
            RF_num = 20 # number of RFdiffusion designs for each template
            MPNN_num = 10 # number of PorteinMPNN designs for each RF pdb
            current_step = 3    # start from current_step

            # STEP 1
            # example: python ./helper_scripts/make_secstruc_adj.py --input_pdb ./2KL8.pdb --out_dir ./adj_ss
            if current_step <= 1:
                selected_pdb_dir = os.path.join(self.tmp_dir, 'pdb')
                os.makedirs(selected_pdb_dir, exist_ok=True)
                # overwrite ss adj dir
                ss_adj_dir = os.path.join(self.intermediate_dir, 'ss_adj')
                self.clean_dirs(ss_adj_dir)
                for pdb_file in tqdm(self.data_list, desc='get test pdb files'):
                    pdb_path = os.path.join(args.data_root, self.benchmark_config.test_set, 'pdb', pdb_file)
                    cp_path = os.path.join(selected_pdb_dir, pdb_file)
                    shutil.copy(pdb_path, cp_path)
                ss_adj_script = os.path.join('baselines', 'RFdiffusion', 'helper_scripts', 'make_secstruc_adj.py')
                command = ['python', ss_adj_script]
                command.extend(['--pdb_dir', str(selected_pdb_dir)])
                command.extend(['--out_dir', str(ss_adj_dir)])

                command = ' '.join(command)
                result = subprocess.run(command, shell=True, encoding="utf-8", check=True)

            # STEP 2
            # example: python ./scripts/run_inference.py inference.output_prefix=./outputs/2KL8 scaffoldguided.scaffoldguided=True scaffoldguided.target_pdb=False scaffoldguided.scaffold_dir=./helper_scripts/adj_ss scaffoldguided.target_ss=./helper_scripts/adj_ss/2KL8_ss.pt scaffoldguided.target_adj=./helper_scripts/adj_ss/2KL8_adj.pt inference.num_designs=2
            rf_pdbs = os.path.join(self.intermediate_dir, 'rf_pdbs')
            os.makedirs(rf_pdbs, exist_ok=True)
            if current_step <= 2:
                # not overwrite
                rfdiffusion_script = os.path.join('baselines', 'RFdiffusion', 'scripts', 'run_inference.py')
                command_fixed = ['python', rfdiffusion_script]
                command_fixed.extend([f'scaffoldguided.scaffoldguided=True'])
                command_fixed.extend([f'scaffoldguided.target_pdb=False'])
                command_fixed.extend([f'scaffoldguided.scaffold_dir={str(ss_adj_dir)}'])

                for pdb_file in tqdm(self.data_list, desc='get RFdiffusion pdb files'):
                    pdb_name = pdb_file.split('.')[0]
                    command_variable = []
                    command_variable.extend([f'inference.output_prefix={str(os.path.join(rf_pdbs, pdb_name))}'])
                    command_variable.extend([f'inference.num_designs={str(RF_num)}'])

                    command = command_fixed + command_variable
                    command = ' '.join(command)
                    result = subprocess.run(command, shell=True, encoding="utf-8", check=False)



            # STEP 3
            if current_step <= 3:
                # copy RFdiffusion generated final pdb files only for ProteinMPNN
                ProteinMPNN_inputs = os.path.join(self.intermediate_dir, 'ProteinMPNN_inputs')
                self.clean_dirs(ProteinMPNN_inputs)
                for pdb_file in os.listdir(rf_pdbs):
                    pdb_file_path = os.path.join(rf_pdbs, pdb_file)
                    if os.path.isfile(pdb_file_path):
                        if pdb_file.split('.')[-1] == 'pdb':
                            shutil.copy(pdb_file_path, ProteinMPNN_inputs)

                ProteinMPNN_outputs = os.path.join(self.intermediate_dir, "ProteinMPNN_outputs")
                ProteinMPNN_script = os.path.join('baselines', 'ProteinMPNN', 'protein_mpnn_run.py')
                command_fixed = ['python', ProteinMPNN_script]
                command_fixed.extend(['--path_to_model_weights', os.path.join('baselines', 'ProteinMPNN', 'ca_model_weights')])
                command_fixed.extend(['--num_seq_per_target', str(MPNN_num)])
                command_fixed.extend(['--ca_only'])
                command_fixed.extend(['--out_folder', ProteinMPNN_outputs.replace("\\", "/").split(":")[-1]])    # proteinMPNN script cannot correctly process windows path

                pdb_files = os.listdir(ProteinMPNN_inputs)
                for pdb_file in tqdm(pdb_files, desc='ProteinMPNN Sample'):
                    pdb_path = os.path.join(ProteinMPNN_inputs, pdb_file)
                    command_variable = ['--pdb_path', pdb_path.replace("\\", "/").split(":")[-1]]

                    command = command_fixed + command_variable
                    command_str = ' '.join(command)
                    print('running command:', command_str)
                    result = subprocess.run(command_str, shell=True, encoding="utf-8", check=True)
                    # result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                # convert ProteinMPNN output fasta
                proteinMPNN_out_path = os.path.join(ProteinMPNN_outputs, 'seqs')
                proteinMPNN_out_files = os.listdir(proteinMPNN_out_path)
                seq_records = dict()
                for proteinMPNN_out_file in proteinMPNN_out_files:
                    rf_pdb_name = proteinMPNN_out_file.split('.')[0]
                    template_name = rf_pdb_name.split('_')[0]
                    if template_name not in seq_records.keys():
                        seq_records[template_name] =  list()
                        # the first record of seq_records[template_name] is the wildtype sequence of template pdb
                        template_pdb_path = os.path.join(args.data_root, self.benchmark_config.test_set, 'pdb', template_name+'.pdb')
                        original_seq = extract_sequence_from_pdb(template_pdb_path)
                        first_record = SeqRecord.SeqRecord(Seq.Seq(original_seq), id = f"{template_name}_original", description="")
                        seq_records[template_name].append(first_record)

                    # read records from ProteinMPNN output
                    file_path = os.path.join(proteinMPNN_out_path, proteinMPNN_out_file)
                    with open(file_path, 'r') as f:
                        records = list(SeqIO.parse(f, "fasta"))
                    assert len(records) == MPNN_num + 1     # records[0] is the sequence of input pdb

                    for i, record in enumerate(records[1:]):
                        record.id = template_name+'_'+str(len(seq_records[template_name])-1)
                        record.description = ''
                        seq_records[template_name].append(record)

                for template_name, records in seq_records.items():
                    for record in records:
                        out_file = record.id + '.fasta'
                        out_path = os.path.join(out_dir, out_file)
                        SeqIO.write(record, out_path, "fasta-2line")
        elif re.match('esm_3', self.model_name):
            from esm.models.esm3 import ESM3
            from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
            os.environ["DISABLE_ITERATIVE_SAMPLING_TQDM"] = "True"
            # This will download the model weights and instantiate the model on your machine.
            model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1")
            # Note: Here I manualy edited esm package to make ESM3 SecondaryStructureTokenizer to "ss3" kind (default is "ss8")

            for pdb_file in tqdm(self.data_list, desc='sample'):
                # pdb_file_path = os.path.join(args.data_root, self.benchmark_config.test_set, 'pdb', pdb_file)
                aa_feature_file_path = os.path.join(args.data_root, self.benchmark_config.test_set, 'aa_feature', pdb_file.replace(".pdb", ".pt"))
                seq_name = pdb_file.split('.')[-2]
                out_path = os.path.join(out_dir, seq_name + '_original.fasta')
                # original_seq = extract_sequence_from_pdb(pdb_file_path)
                aa_feature = torch.load(aa_feature_file_path)
                original_seq = aa_feature['aa_seq']

                with open(out_path, 'w') as f:
                    f.write('>' + seq_name + '_original' + '\n' + original_seq)
                for num in tqdm(range(int(self.benchmark_config.sample_num))):
                    out_path = os.path.join(out_dir, seq_name + '_' + str(num) + '.fasta')
                    ss3_seq = self.get_ss_seq(aa_feature, variable_length=self.model_config.type == "variable_length")
                    protein = ESMProtein(secondary_structure=ss3_seq)
                    # Generate the sequence, then the structure. This will iteratively unmask the sequence track.
                    protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=1))
                    sampled_seq = protein.sequence
                    with open(out_path, 'w') as f:
                        f.write('>' + seq_name + '_' + str(num) + '\n' + sampled_seq)
        else:
            raise ValueError(f'invalid model name: {self.model_name}')

        self.clean_dirs(self.tmp_dir)

    @progress_control
    def esmfold(self, length_filter=10000):
        """
        step2: predict structures with ESMFOLD
        :return:
        """
        print("predicting structures with ESMFOLD...")
        in_dir = os.path.join(self.seq_dir, self.model_name)
        if len(os.listdir(in_dir)) == 0:
            raise Exception('Empty input')
        out_dir = os.path.join(self.esmfold_sructure_dir, self.model_name)
        if not self.fold_continue:
            self.clean_dirs(out_dir)

        # run with subprocess
        # command = ['python -m src.data.esmfold']
        # command.extend(['--fasta_dir', in_dir])
        # command.extend(['--out_dir', out_dir])
        # command = ' '.join(command)
        # print('running command:', command)
        # ret = subprocess.run(command, shell=True, encoding="utf-8", check=True)
        plddt_scores = call_esmfold(fasta_dir=in_dir, out_dir=out_dir, length_filter=length_filter)

        self.organize_files(out_dir)

        # Here is implement of foldability metrics
        esmfold_metrics_out_dir = os.path.join(self.esmfold_metrics_dir, self.model_name)
        os.makedirs(esmfold_metrics_out_dir, exist_ok=True)
        plddt_scores_output = os.path.join(esmfold_metrics_out_dir, 'plddt_scores.json')
        with open(plddt_scores_output, 'w') as f:
            json.dump(plddt_scores, f)

    @progress_control
    def colabfold(self):
        """
        step2: predict structures with COLABFOLD
        :return:
        """
        print("predict structures with COLABFOLD...")
        # run colabfold
        in_dir = os.path.join(self.seq_dir, self.model_name)
        if len(os.listdir(in_dir)) == 0:
            raise Exception('Empty input')
        out_dir = os.path.join(self.colabfold_sructure_dir, self.model_name)
        self.clean_dirs(out_dir)

        command = ['colabfold_batch']
        command.extend([in_dir])
        command.extend([self.tmp_dir])

        # editatble
        command.extend(['--num-recycle', '0'])
        command.extend(['--num-relax', '0'])
        command.extend(['--num-models', '5'])

        command = ' '.join(command)
        print('running command:', command)
        ret = subprocess.run(command, shell=True, encoding="utf-8", check=True)

        # copy files to out_dir
        colabfold = ColabFoldParser(self.tmp_dir)
        plddt_scores = dict()
        for folding_name in colabfold.folding_names:
            plddt_scores[folding_name] = colabfold.get_avg_plddt(folding_name)
            pdb, plddt_plot, pae_plot = colabfold.get_path(folding_name)
            shutil.copy(pdb, os.path.join(out_dir, folding_name+'.pdb'))
            shutil.copy(plddt_plot, out_dir)
            shutil.copy(pae_plot, out_dir)

        self.organize_files(out_dir)

        # empty tmp dir
        self.clean_dirs(self.tmp_dir)
        
        # Here is implement of foldability metrics
        colabfold_metrics_out_dir = os.path.join(self.colabfold_metrics_dir, self.model_name)
        if os.path.exists(colabfold_metrics_out_dir):
            shutil.rmtree(colabfold_metrics_out_dir)
        os.makedirs(colabfold_metrics_out_dir, exist_ok=True)
        plddt_scores_output = os.path.join(colabfold_metrics_out_dir, 'plddt_scores.json')
        with open(plddt_scores_output, 'w') as f:
            json.dump(plddt_scores, f)

    @progress_control
    def plddt_filter(self, top_n=100, folding_model='ESMFOLD'):
        """
        step3: filter colabfold structures with plddt
        :return:
        """
        print(f'plddt filter for {folding_model} structures...')
        if folding_model == 'ESMFOLD':
            in_dir = os.path.join(self.esmfold_sructure_dir, self.model_name)
            metrics_out_dir = os.path.join(self.esmfold_metrics_dir, self.model_name)

        elif folding_model == 'COLABFOLD':
            in_dir = os.path.join(self.colabfold_sructure_dir, self.model_name)
            metrics_out_dir = os.path.join(self.colabfold_metrics_dir, self.model_name)
        else:
            raise Exception('Invalid folding model')

        out_dir = os.path.join(self.plddt_filter_structure_dir, folding_model)
        os.makedirs(out_dir, exist_ok=True)
        out_dir = os.path.join(out_dir, self.model_name)
        if len(os.listdir(in_dir)) == 0:
            raise Exception('Empty input')
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        plddt_scores_output = os.path.join(metrics_out_dir, 'plddt_scores.json')
        with open(plddt_scores_output, 'r') as f:
            plddt_scores = json.load(f)

        # find template names
        template_names = []
        for name in plddt_scores.keys():
            ret = re.match('(\w+)_original', name)
            if ret:
                template_names.append(ret.group(1))

        for template in tqdm(os.listdir(in_dir)):
            output_dir = os.path.join(out_dir, template)
            os.makedirs(output_dir, exist_ok=True)
            plddt_rank = list()
            for file in os.listdir(os.path.join(in_dir, template)):
                file_name = file.split('.')
                if file_name[-1] == 'pdb':
                    if re.search('original', file_name[0]):
                        original_name = file_name[0]
                    else:
                        plddt_rank.append((file_name[0], plddt_scores[file_name[0]]))
            plddt_rank.sort(key=lambda x: x[1], reverse=True)
            selected_names = plddt_rank[:top_n]
            selected_names.append((original_name, plddt_scores[original_name]))
            for file in os.listdir(os.path.join(in_dir, template)):
                for selected_name in selected_names:
                    ret = re.search(selected_name[0], file)
                    if ret:
                        shutil.copy(os.path.join(in_dir, template, file), output_dir)


    @progress_control
    def dssp_annotation(self):
        """
        step3: annotate second structures with DSSP
        :return:
        """
        print('running DSSP annotation...')
        in_dir = os.path.join(self.esmfold_sructure_dir, self.model_name)
        out_dir = os.path.join(self.ss_dir, self.model_name)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        ss_dict = dict()
        templates = os.listdir(in_dir)
        if len(templates) == 0:
            raise Exception('Empty input')
        for template in tqdm(templates):
            template_path = os.path.join(in_dir, template)
            pdb_files = os.listdir(template_path)
            for pdb_file in pdb_files:
                file_path = os.path.join(template_path, pdb_file)
                ss_data = generate_ss2aa(pdb_file=file_path)
                folding_name = pdb_file.split('.')[0]
                template_name = folding_name.split('_')[0]

                if template_name not in ss_dict.keys():
                    ss_dict[template_name] = dict()
                ss_dict[template_name][folding_name] = ss_data
        for template_name in ss_dict.keys():
            torch.save(ss_dict[template_name], os.path.join(out_dir, template_name + '.pt'))

    @ staticmethod
    def organize_files(dir):
        """
        organize files from 'dir/file' to 'dir/template_name/file'
        :param dir:
        :return:
        """
        list_dir = os.listdir(dir)
        files = list()
        for file in list_dir:
            if os.path.isfile(os.path.join(dir, file)):
                files.append(file)
        for file in files:
            ret = re.match('(\w+)_', file)
            template_name = ret.group(1)
            template_path = os.path.join(dir, template_name)
            if not os.path.isdir(template_path):
                os.makedirs(template_path)
            shutil.move(os.path.join(dir, file), template_path)
        
    @ staticmethod
    def organize_files_reverse(in_dir, out_dir):
        """
        reverse organize_files() from 'in_dir/template_name/file' to 'out_dir/file'
        (files in in_dir not changed)
        :param 
        :return:
        """
        list_dir = os.listdir(in_dir)
        for template in list_dir:
            template_path = os.path.join(in_dir, template)
            if os.path.isdir(template_path):
                for file in os.listdir(template_path):
                    shutil.copy(os.path.join(template_path, file), out_dir)

    def load_log(self):
        if os.path.isfile(os.path.join(self.log_dir, self.model_name+'.json')):
            with open(os.path.join(self.log_dir, self.model_name+'.json'), 'r') as f:
                self.log = json.load(f)
            if self.restart:
                print("restart benchmark pipeline...")
                self.log = dict()
            print("progress:", self.log)
        else:
            self.log = dict()


    def save_log(self):
        with open(os.path.join(self.log_dir, self.model_name+'.json'), 'w') as f:
            json.dump(self.log, f)

class MetricsCalculation():
    def __init__(self):
        pass

class MetricsCalculation():
    def __init__(self):
        pass

    @ staticmethod
    def ss_align(seq_1, seq_2, remove_loop=False):

        if remove_loop:
            # ss_alphabet = ['H', 'E', 'C']
            # ss_alphabet_dic = {
            #     "H": "H", "G": "H", "E": "E",
            #     "B": "E", "I": "C", "T": "C",
            #     "S": "C", "L": "C", "-": "C",
            #     "P": "C"
            # }
            letters_to_remove = ['C', 'I', "T", "S", "L", "-", "P"]
            seq_1 = ''.join([char for char in seq_1 if char not in letters_to_remove])
            seq_2 = ''.join([char for char in seq_2 if char not in letters_to_remove])
        if len(seq_1) == 0 or len(seq_2) == 0:
            return 0
        aligner = PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 1
        aligner.mismatch_score = 0

        alignments = aligner.align(seq_1, seq_2)
        return alignments

    @ staticmethod
    def seq_identity(seq_1, seq_2, remove_loop=False):
        """
        sequence identity for second structure seqs
        :param seq_1:
        :param seq_2:
        :param remove_loop:
        :return:
        """
        alignments = MetricsCalculation.ss_align(seq_1, seq_2, remove_loop)
        gaps, mismatch, identities = alignments[0].counts().gaps, alignments[0].counts().mismatches, alignments[0].counts().identities
        identity = identities / (gaps + mismatch + identities) * 100
        return identity

    @ staticmethod
    def seq_statistics(seq, ss3=False, remove_loop=False):
        ss3_alphabet = ['H', 'E', 'C']
        ss8_alphabet = ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'L'] # '-' is replaced by 'L' in annotation
        if remove_loop:
            letters_to_remove = ['C', 'I', "T", "S", "L", "-"]
            seq = ''.join([char for char in seq if char not in letters_to_remove])
        if ss3:
            ss_alphabet = ss3_alphabet
        else:
            ss_alphabet = ss8_alphabet
        seq_statistics = [0] * len(ss_alphabet)
        for char in seq:
            seq_statistics[ss_alphabet.index(char)] += 1
        return seq_statistics

    @ staticmethod
    def aa_statistics(seq):
        extended_protein_letters = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
        aa2index = dict()
        for index, aa in enumerate(extended_protein_letters):
            aa2index[aa] = index
            aa2index[aa.lower()] = index
        statistics = [0] * len(extended_protein_letters)
        for aa in seq:
            if aa not in aa2index.keys():
                break
            statistics[aa2index[aa]] += 1
        return statistics

    @ staticmethod
    def seq_composition_mse(seq_1, seq_2, ss3=False, remove_loop=False):
        """

        :param seq_1: predict seq
        :param seq_2: target seq
        :param ss3: whether input sequences are in ss3 format
        :param remove_loop:
        :return:
        """
        seq_1_statistics = MetricsCalculation.seq_statistics(seq_1, ss3, remove_loop)
        seq_2_statistics = MetricsCalculation.seq_statistics(seq_2, ss3, remove_loop)

        pred_ss = torch.tensor(seq_1_statistics, dtype=torch.float)
        target_ss = torch.tensor(seq_2_statistics, dtype=torch.float)
        if not torch.sum(pred_ss).item()<1:
            pred_ss = pred_ss / torch.sum(pred_ss).item()
        if not torch.sum(target_ss).item()<1:
            target_ss = target_ss / torch.sum(target_ss).item()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            mse = mse_loss(pred_ss, target_ss).item()
        return mse

    @ staticmethod
    def calculate_align_info(predicted_pdb_path, reference_pdb_path):
        cmd = f"TMalign {predicted_pdb_path} {reference_pdb_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stderr:
            print("Error in TMalign:", result.stderr)
            return None

        lines = result.stdout.split("\n")
        tm_score_1, tm_score_2, tm_score = None, None, None
        for line in lines:
            if "Aligned length" in line:
                aligned_length = int(line.split(",")[0].split("=")[1].strip())
                rmsd = float(line.split(",")[1].split("=")[1].strip())
                seq_identity = float(line.split(",")[2].split("=")[-1].strip())
            if "TM-score" in line and "Chain_1" in line:
                tm_score_1 = float(line.split(" ")[1].strip())
            if "TM-score" in line and "Chain_2" in line:
                tm_score_2 = float(line.split(" ")[1].strip())

        if tm_score_1 is not None and tm_score_2 is not None:
            tm_score = (tm_score_1 + tm_score_2) / 2

        align_info = {
            "aligned_length": aligned_length,
            "rmsd": rmsd,
            "seq_identity": seq_identity,
            "tm_score": tm_score,
            "tm_score_1": tm_score_1,
            "tm_score_2": tm_score_2
        }
        return align_info


class BenchmarkPipeline(DesignPipeline, MetricsCalculation):
    def __init__(self, model_config, restart=False, benchmark_config=BenchmarkConfig(), fold_continue=False):
        super().__init__(model_config=model_config, restart=restart, benchmark_config=benchmark_config, fold_continue=fold_continue)

    def run(self, length_filter=10000):
        super().run(length_filter=length_filter)
        self.foldability()
        self.designability()
        self.novelty()
        self.variance()
        self.diversity()

    def design(self, plddt_threshold=0.7):
        super().run()
        self.designability()
        self.design_filter(plddt_threshold=plddt_threshold)


    def progress_control(pipeline_process):
        return DesignPipeline.progress_control(pipeline_process)

    @progress_control
    def foldability(self, folding_model='ESMFOLD'):
        """
        plddt of generated sequence
        plddt calculation implemented in colabfold() & esmfold() of class DesignPipeline
        :return:
        """
        print(f'Foldability Calculation...')
        if folding_model == 'ESMFOLD':
            metrics_out_dir = os.path.join(self.esmfold_metrics_dir, self.model_name)
        elif folding_model == 'COLABFOLD':
            metrics_out_dir = os.path.join(self.colabfold_metrics_dir, self.model_name)
        else:
            raise Exception('Invalid folding model')

        avg_plddt_list = list()
        plddt_scores_file = os.path.join(metrics_out_dir, 'plddt_scores.json')
        avg_plddt_scores_file = os.path.join(metrics_out_dir, 'avg_plddt_scores.json')
        with open(plddt_scores_file, 'r') as f:
            plddt_scores = json.load(f)
        for folding, plddt in plddt_scores.items():
            avg_plddt_list.append(plddt)
        avg_plddt_score = np.array(avg_plddt_list).mean().item()
        avg_plddt = {'avg_plddt': avg_plddt_score}
        print(f'Average plddt: {avg_plddt_score}')
        with open(avg_plddt_scores_file, 'w') as f:
            json.dump(avg_plddt, f)

    @progress_control
    def designability(self):
        """
        check the identity at the second structure level
        :return:
        """
        print('Designability Calculating...')
        in_dir = os.path.join(self.ss_dir, self.model_name)
        out_dir = os.path.join(self.seq_metrics_dir, self.model_name)
        if len(os.listdir(in_dir)) == 0:
            raise Exception(f'Empty input: {in_dir}')
        self.clean_dirs(out_dir)

        def calculate_designability(in_dir, remove_loop=False):
            print(f'Designability (remove_loop={remove_loop})')
            ss_type_nums = [3, 8]
            designability = dict()
            designability_raw = dict()
            # used to draw pie chart
            statistics = ['target_statistics', 'original_statistics', 'design_statistics']
            composition_statistics_raw = {'global': {}, 'local': {}}
            for ss_type_num in ss_type_nums:
                composition_statistics_raw['global'][ss_type_num] = dict()
                for variable_name in statistics:
                    composition_statistics_raw['global'][ss_type_num][variable_name] = [0]*ss_type_num
            global_target3_statistics = [0] * 3
            global_original3_statistics = [0] * 3
            global_target8_statistics = [0] * 8
            global_original8_statistics = [0] * 8
            global_design3_statistics = [0] * 3
            global_design8_statistics = [0] * 8
            for template_file in tqdm(os.listdir(in_dir), desc="designability"):
                try:
                    template = template_file.split('.')[0]
                    designability[template] = dict()
                    designability_raw[template] = dict()
                    composition_statistics_raw['local'][template] = dict()
                    designed_ss_dict = torch.load(os.path.join(in_dir, template_file))
                    input_ss = torch.load(os.path.join('data', self.benchmark_config.test_set, 'aa_feature', template_file))
                    ss3_identity_list = list()
                    ss8_identity_list = list()
                    ss3_composition_mse_list = list()
                    ss8_composition_mse_list = list()

                    for designed_name, designed_ss in designed_ss_dict.items():
                        designability_raw[template][designed_name] = dict()
                        composition_statistics_raw['local'][template][designed_name] = dict()
                        for ss_type_num in ss_type_nums:
                            designability_raw[template][designed_name][ss_type_num] = dict()
                            composition_statistics_raw['local'][template][designed_name][ss_type_num] = dict()
                            designability_raw[template][designed_name][ss_type_num]['identity'] = self.seq_identity(designed_ss[f"ss{ss_type_num}_seq"], input_ss[f"ss{ss_type_num}_seq"], remove_loop=remove_loop)
                            designability_raw[template][designed_name][ss_type_num]['composition_mse'] = self.seq_composition_mse(designed_ss[f"ss{ss_type_num}_seq"], input_ss[f"ss{ss_type_num}_seq"], ss3=True if ss_type_num == 3 else False, remove_loop=remove_loop)
                        if designed_name == template+'_original':
                            composition_statistics_raw['local'][template][template+'_target'] = dict()
                            ss3_identity_original = self.seq_identity(designed_ss["ss3_seq"], input_ss["ss3_seq"], remove_loop=remove_loop)
                            ss8_identity_original = self.seq_identity(designed_ss["ss8_seq"], input_ss["ss8_seq"], remove_loop=remove_loop)
                            ss3_composition_mse_original = self.seq_composition_mse(designed_ss["ss3_seq"], input_ss["ss3_seq"], ss3=True, remove_loop=remove_loop)
                            ss8_composition_mse_original = self.seq_composition_mse(designed_ss["ss8_seq"], input_ss["ss8_seq"], ss3=False, remove_loop=remove_loop)

                            # local_target3_statistics = self.seq_statistics(input_ss["ss3_seq"], ss3=True, remove_loop=remove_loop)
                            # local_original3_statistics = self.seq_statistics(designed_ss["ss3_seq"], ss3=True, remove_loop=remove_loop)
                            # local_target8_statistics = self.seq_statistics(input_ss["ss8_seq"], ss3=False, remove_loop=remove_loop)
                            # local_original8_statistics = self.seq_statistics(designed_ss["ss8_seq"], ss3=False, remove_loop=remove_loop)
                            composition_statistics_raw['local'][template][template+'_target'][3] = self.seq_statistics(input_ss["ss3_seq"], ss3=True, remove_loop=remove_loop)
                            composition_statistics_raw['local'][template][template+'_original'][3] = self.seq_statistics(designed_ss["ss3_seq"], ss3=True, remove_loop=remove_loop)
                            composition_statistics_raw['local'][template][template+'_target'][8] = self.seq_statistics(input_ss["ss8_seq"], ss3=False, remove_loop=remove_loop)
                            composition_statistics_raw['local'][template][template+'_original'][8] = self.seq_statistics(designed_ss["ss8_seq"], ss3=False, remove_loop=remove_loop)

                            # global_target3_statistics = [x + y for x, y in zip(global_target3_statistics, local_target3_statistics)]
                            # global_original3_statistics = [x + y for x, y in zip(global_original3_statistics, local_original3_statistics)]
                            # global_target8_statistics = [x + y for x, y in zip(global_target8_statistics, local_target8_statistics)]
                            # global_original8_statistics = [x + y for x, y in zip(global_original8_statistics, local_original8_statistics)]

                        else:
                            ss3_identity_list.append(self.seq_identity(designed_ss["ss3_seq"], input_ss["ss3_seq"], remove_loop=remove_loop))
                            ss8_identity_list.append(self.seq_identity(designed_ss["ss8_seq"], input_ss["ss8_seq"], remove_loop=remove_loop))
                            ss3_composition_mse_list.append(self.seq_composition_mse(designed_ss["ss3_seq"], input_ss["ss3_seq"], ss3=True, remove_loop=remove_loop))
                            ss8_composition_mse_list.append(self.seq_composition_mse(designed_ss["ss8_seq"], input_ss["ss8_seq"], ss3=False, remove_loop=remove_loop))

                            # local_design3_statistics = self.seq_statistics(designed_ss["ss3_seq"], ss3=True, remove_loop=remove_loop)
                            # local_design8_statistics = self.seq_statistics(designed_ss["ss8_seq"], ss3=False, remove_loop=remove_loop)
                            composition_statistics_raw['local'][template][designed_name][3] = self.seq_statistics(designed_ss["ss3_seq"], ss3=True, remove_loop=remove_loop)
                            composition_statistics_raw['local'][template][designed_name][8] = self.seq_statistics(input_ss["ss8_seq"], ss3=False, remove_loop=remove_loop)


                            # global_design3_statistics = [x + y for x, y in zip(global_design3_statistics, local_design3_statistics)]
                            # global_design8_statistics = [x + y for x, y in zip(global_design8_statistics, local_design8_statistics)]

                    ss3_identity = np.array(ss3_identity_list).mean().item()
                    ss8_identity = np.array(ss8_identity_list).mean().item()
                    ss3_identity_max = np.max(np.array(ss3_identity_list)).item()
                    ss8_identity_max = np.max(np.array(ss8_identity_list)).item()
                    ss3_composition_mse = np.array(ss3_composition_mse_list).mean().item()
                    ss8_composition_mse = np.array(ss8_composition_mse_list).mean().item()
                    ss3_identity_list.sort(reverse=True)
                    ss8_identity_list.sort(reverse=True)
                    ss3_identity_top10 = np.array(ss3_identity_list[:10]).mean().item()
                    ss8_identity_top10 = np.array(ss8_identity_list[:10]).mean().item()



                    # designability_metrics = ['ss3_identity', 'ss3_identity_max', 'ss3_identity_top10', 'ss3_identity_original', 'ss3_composition_mse', 'ss3_composition_mse_original',
                    #                          'ss8_identity', 'ss8_identity_max', 'ss8_identity_top10', 'ss8_identity_original', 'ss8_composition_mse', 'ss8_composition_mse_original']
                    # for designability_metric in designability_metrics:
                    #     designability[template][designability_metric] = eval(designability_metric)
                    designability[template]['ss3_identity'] = ss3_identity
                    designability[template]['ss8_identity'] = ss8_identity
                    designability[template]['ss3_identity_max'] = ss3_identity_max
                    designability[template]['ss8_identity_max'] = ss8_identity_max
                    designability[template]['ss3_identity_top10'] = ss3_identity_top10
                    designability[template]['ss8_identity_top10'] = ss8_identity_top10
                    designability[template]['ss3_identity_original'] = ss3_identity_original
                    designability[template]['ss8_identity_original'] = ss8_identity_original
                    designability[template]['ss3_composition_mse'] = ss3_composition_mse
                    designability[template]['ss8_composition_mse'] = ss8_composition_mse
                    designability[template]['ss3_composition_mse_original'] = ss3_composition_mse_original
                    designability[template]['ss8_composition_mse_original'] = ss8_composition_mse_original
                    # debug
                    print(designability[template])
                    # print(template, 'mse', ss3_composition_mse)
                except Exception as e:
                    print("error:", e)
                    template = template_file.split('.')[0]
                    designability.pop(template, None)
                    designability_raw.pop(template, None)
                    composition_statistics_raw['local'].pop(template, None)

            # claculate global_composition_statistics:
                for template_name, template in composition_statistics_raw['local'].items():
                    for design_name, design in template.items():
                        for ss_type_num, statistics in design.items():
                            if design_name == template_name+"_original":
                                composition_statistics_raw['global'][ss_type_num]['original_statistics'] = [x + y for x, y in zip(composition_statistics_raw['global'][ss_type_num]['original_statistics'], statistics)]
                            elif design_name == template_name+"_target":
                                composition_statistics_raw['global'][ss_type_num]['target_statistics'] = [x + y for x, y in zip(composition_statistics_raw['global'][ss_type_num]['target_statistics'], statistics)]
                            else:
                                composition_statistics_raw['global'][ss_type_num]['design_statistics'] = [x + y for x, y in zip(composition_statistics_raw['global'][ss_type_num]['design_statistics'], statistics)]


            # save composition_statistics_raw
            with open(os.path.join(out_dir, f"composition_statistics_raw{'_remove_loop' if remove_loop else ''}.json"), 'w') as f:
                json.dump(composition_statistics_raw, f)

            # if draw_pie_chart:
            #     statistics = [
            #         ['global_target3_statistics', 'global_original3_statistics', 'global_design3_statistics'],
            #         ['global_target8_statistics', 'global_original8_statistics', 'global_design8_statistics']]
            #     labels = [['H', 'E', 'C'], ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'L']]
            #     ss_statistics_fig_path = os.path.join(out_dir, 'figure')
            #     os.makedirs(ss_statistics_fig_path, exist_ok=True)
            #     for i, ss_i_statistics in enumerate(statistics):
            #         fig, axs = plt.subplots(3)
            #         for j, variable in enumerate(ss_i_statistics):
            #             print(variable, eval(variable))
            #             axs[j].pie(eval(variable), labels=labels[i],
            #                     textprops={'color': 'b', 'size': 10, 'weight': 'bold'})
            #             axs[j].set_title(f"{variable}(remove_loop={remove_loop})", c='b')
            #         plt.savefig(os.path.join(ss_statistics_fig_path, f"ss_statistics_{str(i)}{'_remove_loop'if remove_loop else ''}.png"))
            #         plt.cla()

            new_avg_list = dict()
            avg_list = dict()
            for template, data in designability.items():
                for data_type, data_num in data.items():
                    if data_type not in avg_list.keys():
                        avg_list[data_type] = list()
                    if not np.isnan(data_num):
                        avg_list[data_type].append(data_num)
            # debug
            # print(avg_list['ss3_composition_mse'])
            for data_type in avg_list.keys():
                new_avg_list[data_type] = np.array(avg_list[data_type]).mean().item()
                new_avg_list[data_type+'_std'] = np.array(avg_list[data_type]).std().item()
                print(data_type, new_avg_list[data_type])
                print(data_type+'_std', new_avg_list[data_type+'_std'])
            return designability_raw, designability, new_avg_list

        designability_raw, designability, avg_list = calculate_designability(in_dir, False)
        designability_raw_remove_loop, designability_remove_loop, avg_list_remove_loop = calculate_designability(in_dir, remove_loop=True)
            
        with open(os.path.join(out_dir, 'designability.json'), 'w') as f:
            json.dump(designability, f)
        with open(os.path.join(out_dir, 'avg_designability.json'), 'w') as f:
            json.dump(avg_list, f)
        with open(os.path.join(out_dir, 'designability_raw.json'), 'w') as f:
            json.dump(designability_raw, f)
        with open(os.path.join(out_dir, 'designability_raw_remove_loop.json'), 'w') as f:
            json.dump(designability_raw_remove_loop, f)
        with open(os.path.join(out_dir, 'designability_remove_loop.json'), 'w') as f:
            json.dump(designability_remove_loop, f)
        with open(os.path.join(out_dir, 'avg_designability_remove_loop.json'), 'w') as f:
            json.dump(avg_list_remove_loop, f)

        # draw pie chart with matplotlib.pyplot (global statistics: statistics across all templates)
        # self.draw_ss_statistics()

    def diversity(self, folding_model='ESMFOLD'):
        """
        the number of structural clusters by MaxCluster (structural clusters or sequential clusters?
        :return:
        """
        print('Diversity Calculating...')
        if folding_model == 'ESMFOLD':
            in_dir = os.path.join(self.esmfold_sructure_dir, self.model_name)
            out_dir = os.path.join(self.esmfold_metrics_dir, self.model_name)
        elif folding_model == 'COLABFOLD':
            in_dir = os.path.join(self.colabfold_sructure_dir, self.model_name)
            out_dir = os.path.join(self.colabfold_metrics_dir, self.model_name)
        else:
            raise Exception('Invalid folding model')
        if len(os.listdir(in_dir)) == 0:
            raise Exception('Empty input')
        os.makedirs(out_dir, exist_ok=True)
            
        diversity = dict()
        templates = os.listdir(in_dir)
        for template in tqdm(templates):
            try:
                if os.path.isdir(os.path.join(in_dir, template)):
                    diversity[template] = dict()
                    gen_file_path = list()
                    mutual_TM = list()
                    mutual_rmsd = list()
                    mutual_seq_identity = list()
                    original_TM = list()
                    original_rmsd = list()
                    original_seq_identity = list()
                    for file in os.listdir(os.path.join(in_dir, template)):
                        ret1 = re.match('\S+\.pdb', file)
                        if ret1:
                            file = ret1.group(0)
                            ret2 = re.search('original', file)
                            if ret2:
                                ori_file_path = os.path.join(in_dir, template, file)
                            else:
                                gen_file_path.append(os.path.join(in_dir, template, file))
                    for i in range(len(gen_file_path)):
                        for j in range(len(gen_file_path)):
                            if i < j:
                                aligned_info = self.calculate_align_info(gen_file_path[i], gen_file_path[j])
                                if aligned_info:
                                    mutual_TM.append(aligned_info['tm_score'])
                                    mutual_rmsd.append(aligned_info['rmsd'])
                                    mutual_seq_identity.append(aligned_info["seq_identity"])
                    for i in range(len(gen_file_path)):
                        aligned_info = self.calculate_align_info(gen_file_path[i], ori_file_path)
                        if aligned_info:
                            original_TM.append(aligned_info['tm_score'])
                            original_rmsd.append(aligned_info['rmsd'])
                            original_seq_identity.append(aligned_info["seq_identity"])

                    diversity[template]['mutual_TM'] = np.array(mutual_TM).mean().item()
                    diversity[template]['mutual_rmsd'] = np.array(mutual_rmsd).mean().item()
                    diversity[template]['mutual_seq_identity'] = np.array(mutual_seq_identity).mean().item()
                    diversity[template]['TM2original'] = np.array(original_TM).mean().item()
                    diversity[template]['rmsd2original'] = np.array(original_rmsd).mean().item()
                    diversity[template]['seq_identity2original'] = np.array(original_seq_identity).mean().item()
            except Exception as e:
                print("error:", e)
                diversity.pop(template, None)

        new_avg_list = dict()
        avg_list = dict()
        for template, data in diversity.items():
            for data_type, data_num in data.items():
                if data_type not in avg_list.keys():
                    avg_list[data_type] = list()
                if not np.isnan(data_num):
                    avg_list[data_type].append(data_num)
        for data_type in avg_list.keys():
            new_avg_list[data_type] = np.array(avg_list[data_type]).mean().item()
            new_avg_list[data_type + '_std'] = np.array(avg_list[data_type]).std().item()
            print(data_type, new_avg_list[data_type])
            print(data_type + '_std', new_avg_list[data_type + '_std'])
        with open(os.path.join(out_dir, 'diversity.json'), 'w') as f:
            json.dump(diversity, f)
        with open(os.path.join(out_dir, 'avg_diversity.json'), 'w') as f:
            json.dump(new_avg_list, f)

    @progress_control
    def novelty(self, folding_model='ESMFOLD'):
        """
        pdbTM: The highest average TM-score of designable samples to any chain in training set of natural backbones.
        The similar structures are searched by FoldSeek.
        :return:
        """
        print('Novelty Calculating...')
        if folding_model == 'ESMFOLD':
            query_dataset = os.path.join(self.esmfold_sructure_dir, self.model_name)
            out_dir = os.path.join(self.esmfold_metrics_dir, self.model_name)
        elif folding_model == 'COLABFOLD':
            query_dataset = os.path.join(self.colabfold_sructure_dir, self.model_name)
            out_dir = os.path.join(self.colabfold_metrics_dir, self.model_name)
        else:
            raise Exception('Invalid folding model')
        if len(os.listdir(query_dataset)) == 0:
            raise Exception('Empty input')
        os.makedirs(out_dir, exist_ok=True)

        # find dirs of query & target pdb files
        query_pdb = os.path.join(self.foldseek_dir, 'query_pdb')
        os.makedirs(query_pdb, exist_ok=True)
        self.organize_files_reverse(in_dir=query_dataset, out_dir=query_pdb)
        target_pdb = os.path.join(args.data_root, self.benchmark_config.train_set, 'pdb')

        queryDB = os.path.join(self.foldseek_dir, 'queryDB')
        targetDB = os.path.join(self.foldseek_dir, 'targetDB')

        foldseek_path = os.path.join('script', 'shell', 'foldseek.sh')
        aln_tsv = os.path.join(out_dir, 'aln_tmscore.tsv')
        command = f"bash {foldseek_path} {query_pdb} {target_pdb} {queryDB} {targetDB} {self.foldseek_dir} {aln_tsv}"
        print("running command:", command)
        result = subprocess.run(command, shell=True, encoding="utf-8", check=True)
        if result.stderr:
            print("Error in Foldseek:", result.stderr)

        align_closest = dict()
        avg_TM_score_list = list()
        with open(aln_tsv, 'r', encoding='utf-8') as tsvfile:
            lines = tsvfile.readlines()
            for line in lines:
                tab_split = line.split('\t')
                space_split = tab_split[1].split(' ')
                query_file = tab_split[0]
                query = query_file.split('.')[0]
                target_file = space_split[0]
                target = target_file.split('.')[0]
                tm_score = float(space_split[1])
                if query not in align_closest.keys():
                    align_closest[query] = {'target': target, 'tm_score': tm_score}
                else:
                    if align_closest[query]['tm_score'] > tm_score:
                        align_closest[query] = {'target': target, 'tm_score': tm_score}
        for query, info in align_closest.items():
            avg_TM_score_list.append(info['tm_score'])
        with open(os.path.join(out_dir, 'novelty_align.json'), 'w') as f:
            json.dump(align_closest, f)
        with open(os.path.join(out_dir, 'novelty.json'), 'w') as f:
            avg_TM_score = np.array(avg_TM_score_list).mean().item()
            avg_TM_score_std = np.array(avg_TM_score_list).std().item()
            print('average TM score between generated structure and its closest strcuture in training set: ', avg_TM_score, avg_TM_score_std)
            json.dump({'avg_TM_score': avg_TM_score, 'avg_TM_score_std': avg_TM_score_std}, f)
        self.clean_dirs(self.foldseek_dir)

    def designability_plot(self):
        print('Designability: ')
        in_dir = os.path.join(self.seq_metrics_dir, self.model_name)
        designability_path = os.path.join(in_dir, 'designability.json')
        with open(designability_path, 'r') as f:
            designability = json.load(f)
        ss3_identity_list = list()
        ss8_identity_list = list()
        ss3_identity_original_list = list()
        ss8_identity_original_list = list()
        for template, identity in designability.items():
            for identity_type, identity_num in identity.items():
                if not np.isnan(identity_num):
                    eval(identity_type+'_list').append(identity_num)
        print('ss3_identity', np.array(ss3_identity_list).mean().item())
        print('ss8_identity', np.array(ss8_identity_list).mean().item())
        print('ss3_identity_original', np.array(ss3_identity_original_list).mean().item())
        print('ss8_identity_original', np.array(ss8_identity_original_list).mean().item())
        
    def design_filter(self, top_n=50, plddt_threshold=0.7, folding_model='ESMFOLD', ss_type_num=3):
        designability_in_dir = os.path.join(self.seq_metrics_dir, self.model_name, 'designability_raw.json')
        with open(designability_in_dir, 'r') as f:
            designability = json.load(f)
        print(f'plddt filter for {folding_model} structures...')
        if folding_model == 'ESMFOLD':
            in_dir = os.path.join(self.esmfold_sructure_dir, self.model_name)
            metrics_out_dir = os.path.join(self.esmfold_metrics_dir, self.model_name)
        elif folding_model == 'COLABFOLD':
            in_dir = os.path.join(self.colabfold_sructure_dir, self.model_name)
            metrics_out_dir = os.path.join(self.colabfold_metrics_dir, self.model_name)
        else:
            raise Exception('Invalid folding model')
        plddt_scores_output = os.path.join(metrics_out_dir, 'plddt_scores.json')
        with open(plddt_scores_output, 'r') as f:
            plddt_scores = json.load(f)

        if len(os.listdir(in_dir)) == 0:
            raise Exception('Empty input')
        out_dir = os.path.join(self.design_filter_structure_dir, self.model_name)
        self.clean_dirs(out_dir)

        global_design_list = list()
        for template in tqdm(designability.keys()):
            output_dir = os.path.join(out_dir, template)
            os.makedirs(output_dir, exist_ok=True)
            design_list = list()
            for design in designability[template].keys():
                # plddt filter
                if plddt_scores[design] > plddt_threshold:
                    design_list.append((design, designability[template][design][str(ss_type_num)]['identity'], plddt_scores[design], designability[template][design][str(ss_type_num)]['composition_mse']))
            global_design_list.extend(design_list)

            design_list.sort(key=lambda x: x[1], reverse=True)
            # write ranking & copy pdb_file for every template
            with open(os.path.join(output_dir, 'ranking.csv'), 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['name', 'identity', 'plddt', 'composition_mse'])
                for i in range(min(top_n, len(design_list))):
                    writer.writerow(design_list[i])
                    shutil.copy(os.path.join(in_dir, template, design_list[i][0]+'.pdb'), output_dir)

        # delete original seq from global ranking
        global_design_list = [data for data in global_design_list if not re.match(".+original$", data[0])]
        # write global ranking
        global_design_list.sort(key=lambda x: x[1], reverse=True)
        with open(os.path.join(out_dir, 'global_ranking.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['name', 'identity', 'plddt', 'composition_mse'])
            for i in range(len(global_design_list)):
                writer.writerow(global_design_list[i])


    @progress_control
    def variance(self):
        """
        variance on length and sequence identity
        :return:
        """
        pass

    def over_representation(self):
        """
        A Non-Parametric Test to Detect Data-Copying in Generative Models (AISTATS, 2020) https://ucsdml.github.io/jekyll/update/2020/08/03/how-to-detect-data-copying-in-generative-models.html
        :return:
        """
        pass

    def data_copying(self):
        pass


class BenchmarkAnalysis:
    def __init__(self, model_configs, args, benchmark_config=BenchmarkConfig()):
        self.args = args
        self.Baselines = dict()
        for model_config in model_configs:
            baseline = BenchmarkPipeline(model_config=model_config, benchmark_config=benchmark_config)
            self.Baselines[self.get_abbreviation(baseline.model_name)] = baseline
        self.outdir = os.path.join(list(self.Baselines.values())[0].base_dir, 'analysis_figure')

        os.makedirs(self.outdir, exist_ok=True)

    @staticmethod
    def get_abbreviation(baseline_names):

        def get_abbrev(name):
            baseline_abbreviation = {'protdiff_2nd_S40': "CPDiffusion-SS", "esm_if1_gvp4_t16_142M_UR50": "ESM-IF1",
                                     'prostT5': "ProstT5", "ProteinMPNN": "ProteinMPNN",
                                     "esm_2": "ESM-2(1)", "esm_2_ratio_06": "ESM-2(0.6)",
                                     "esm_2_ratio_08": "ESM-2(0.8)",
                                     "esm_2_PLM": "ESM-2(1)", 'prostT5_foldseek_PLM': "ProstT5",
                                     "ProteinMPNN_IF": "ProteinMPNN", "esm_if1_gvp4_t16_142M_UR50_IF": "ESM-IF1"}
            if name in baseline_abbreviation.keys():
                return baseline_abbreviation[name]
            else:
                return name

        if type(baseline_names) == str:
            return get_abbrev(baseline_names)
        else:
            return [get_abbrev(name) for name in baseline_names]


    def ss_composition(self, template_name=None, design_names=None, case_study=None):
        """
        draw pie chart of global (acorss all templates) or local (one template) or case (each design) statistics
        compute avg ss_composition of each baseline over all templates

        global statistics params
        :param template_name: template name for local ss_composition calculation. Set None for global ss_composition calculation
        :param design_names(optional): design names for local ss_composition calculation. List of design_name (set None to use default design names, which is ['WT', 'design_avg', 'WT_seq', ...])
        case statistics params
        :param case_study: [(baseline_name, design_name), ...]
        :return:
        """
        print(f"drawing ss composition...")
        # plot settings
        ss_type_nums = ["3", "8"]
        labels = [['H', 'E', 'C'], ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'L']]
        # color_labels = [['#1f77b4', '#ff7f0e', '#2ca02c'],
        #                 ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']]
        color_labels = [['#d3c6b3', '#a7adba', '#b2beb5'],
                        ['#d3c6b3', '#a7adba', '#b2beb5', '#c0c0c0',
                        '#d8bfd8', '#e3dac9', '#f0ead6', '#c4aead']]
        # TODO
        color_labels[0][0] = np.array([176 / 255, 52 / 255, 60 / 255])
        color_labels[0][1] = np.array([210 / 255, 170 / 255, 87 / 255])
        color_labels[0][2] = np.array([147 / 255, 168 / 255, 178 / 255])


        case_outdir = os.path.join(self.outdir, 'case_study')
        os.makedirs(case_outdir, exist_ok=True)

        for remove_loop in [True, False]:
            baseline_composition_statistics = {"3": {}, "8": {}}
            case_study_statistics = {"3": {}, "8": {}}
            case_study_name = ''    # template name for this case study figure
            # draw global pie chart & accumulative histogram for each baseline
            for baseline in self.Baselines.values():
                baseline_out_dir = os.path.join(self.outdir, baseline.model_name)
                os.makedirs(baseline_out_dir, exist_ok=True)
                in_dir = os.path.join(baseline.seq_metrics_dir, baseline.model_name)
                file_dir = os.path.join(in_dir, f"composition_statistics_raw{'_remove_loop' if remove_loop else ''}.json")
                if not os.path.isfile(file_dir):
                    baseline.rerun(baseline.designability)
                with open(file_dir, 'r') as f:
                    composition_statistics_raw = json.load(f)
                ss_statistics_fig_path = baseline_out_dir
                if template_name is not None:
                    if template_name in composition_statistics_raw['local'].keys():
                        ss_statistics_fig_path = os.path.join(ss_statistics_fig_path, template_name)
                        os.makedirs(ss_statistics_fig_path, exist_ok=True)
                        statistics_names = ['design_avg']
                        template_statistics = dict()
                        for ss_type_num in ss_type_nums:    # init
                            template_statistics[ss_type_num] = dict()
                            for variable_name in statistics_names:
                                template_statistics[ss_type_num][variable_name] = [0] * int(ss_type_num)

                        # claculate template_composition_statistics:
                        for design_name, design in composition_statistics_raw['local'][template_name].items():
                            for ss_type_num, statistics in design.items():
                                if design_name == template_name+"_original":
                                    template_statistics[ss_type_num]['WT_seq'] = statistics
                                elif design_name == template_name+"_target":
                                    template_statistics[ss_type_num]['WT'] = statistics
                                else:
                                    template_statistics[ss_type_num]['design_avg'] = [x + y for x, y in zip(template_statistics[ss_type_num]['design_avg'], statistics)]
                                    template_statistics[ss_type_num][design_name] = statistics

                        # calculate percentage of each ss composition
                        for ss_type_num, ss_statistics in template_statistics.items():
                            for name, statistics in ss_statistics.items():
                                ss_percentage = torch.tensor(statistics, dtype=torch.float)
                                if not torch.sum(ss_percentage).item() < 1:
                                    ss_percentage = ss_percentage / torch.sum(ss_percentage).item()
                                template_statistics[ss_type_num][name] = ss_percentage.numpy()

                        # draw accumulative histogram
                        for i, ss_type_num in enumerate(ss_type_nums):
                            if design_names is None:
                                design_names = ['WT']+list(template_statistics[ss_type_num].keys())[:5]
                            ss_type_percentage = dict()
                            bottom = [0]*len(design_names)
                            for ss_type_index in range(int(ss_type_num)):
                                ss_type_percentage[ss_type_index] = [template_statistics[ss_type_num][name][ss_type_index].item() for name in design_names]
                                plt.bar(design_names, ss_type_percentage[ss_type_index], width=0.4, bottom=bottom, label=labels[i][ss_type_index], color=color_labels[i][ss_type_index], edgecolor='grey', zorder=5)
                                bottom = [x + y for x, y in zip(bottom, ss_type_percentage[ss_type_index])]


                            plt.tick_params(axis='x', length=0)
                            plt.xlabel('Name', fontsize=12)
                            plt.ylabel('ss composition', fontsize=12)
                            plt.ylim(0, 1.01)
                            plt.yticks(np.arange(0, 1.2, 0.2), [f'{i}%' for i in range(0, 120, 20)])
                            plt.grid(axis='y', alpha=0.5, ls='--')
                            plt.legend(frameon=False, bbox_to_anchor=(1.01, 1))
                            plt.tight_layout()
                            plt.savefig(os.path.join(ss_statistics_fig_path,
                                                     f"{template_name}_ss{str(ss_type_num)}_statistics_{'_remove_loop' if remove_loop else ''}.png"))
                            plt.cla()
                    else:
                        print(f"Unable to draw ss composition pie chart for template {template_name}: no such template in {file_dir}")
                elif case_study is not None:
                    for (baseline_abbre, design_name) in case_study:
                        if baseline_abbre == self.get_abbreviation(baseline.model_name):
                            ret = re.search('(\w+)_', design_name)
                            template = ret.group(1)
                            # Note: here consider each baseline only have one template
                            if case_study_name == '':
                                case_study_name = template

                            for ss_type_num in ss_type_nums:
                                if 'WT' not in case_study_statistics[ss_type_num].keys():   # get WT statistics
                                    WT_path = os.path.join(self.args.data_root, baseline.benchmark_config.test_set,
                                                           "aa_feature", f"{template}.pt")
                                    WT_ss = torch.load(WT_path)
                                    # calculate ss statistics
                                    case_study_statistics[ss_type_num]['WT'] = MetricsCalculation.seq_statistics(WT_ss["ss"+ss_type_num+"_seq"], ss3=ss_type_num == '3', remove_loop=remove_loop)

                                # Note: here consider each baseline only have one design
                                design_path = os.path.join(baseline.ss_dir, baseline.model_name, f"{template}.pt")
                                design_ss = torch.load(design_path)[design_name]
                                case_study_statistics[ss_type_num][self.get_abbreviation(baseline.model_name)] = MetricsCalculation.seq_statistics(design_ss["ss"+ss_type_num+"_seq"], ss3=ss_type_num == '3', remove_loop=remove_loop)
                else:   # global ss_composition calculation
                    for ss_type_num in ss_type_nums:
                        if 'WT' not in baseline_composition_statistics[ss_type_num].keys():
                            baseline_composition_statistics[ss_type_num]['WT'] = composition_statistics_raw['global'][ss_type_num]['target_statistics']
                        if 'WT_seq' not in baseline_composition_statistics[ss_type_num].keys():
                            baseline_composition_statistics[ss_type_num]['WT_seq'] = composition_statistics_raw['global'][ss_type_num]['original_statistics']
                        baseline_composition_statistics[ss_type_num][baseline.model_name] = composition_statistics_raw['global'][ss_type_num]['design_statistics']

                    # draw global pie chart
                    for i, (ss_type_num, ss_statistics) in enumerate(composition_statistics_raw['global'].items()):
                        fig, axs = plt.subplots(3)
                        for j, (statistics_name, statistics) in enumerate(ss_statistics.items()):
                            variable_name = f"global_ss{str(ss_type_num)}_{statistics_name}"
                            print(variable_name, statistics)
                            axs[j].pie(statistics, labels=labels[i],
                                       textprops={'color': 'b', 'size': 10, 'weight': 'bold'})
                            axs[j].set_title(f"{variable_name}(remove_loop={remove_loop})", c='b')
                        plt.savefig(os.path.join(ss_statistics_fig_path,
                                                 f"ss{str(ss_type_num)}_statistics_{'_remove_loop' if remove_loop else ''}.png"))
                        plt.cla()



            plt.figure()
            if template_name is not None:
                pass
            elif case_study is not None:
                print(case_study_statistics)
                # calculate percentage of each ss composition
                for ss_type_num, ss_statistics in case_study_statistics.items():
                    for name, statistics in ss_statistics.items():
                        ss_percentage = torch.tensor(statistics, dtype=torch.float)
                        if not torch.sum(ss_percentage).item() < 1:
                            ss_percentage = ss_percentage / torch.sum(ss_percentage).item()
                        case_study_statistics[ss_type_num][name] = ss_percentage.numpy()
                # draw case study accumulative histogram
                for i, ss_type_num in enumerate(ss_type_nums):
                    baseline_names = case_study_statistics["3"].keys()
                    ss_type_percentage = dict()
                    bottom = [0] * len(baseline_names)
                    for ss_type_index in range(int(ss_type_num)):
                        ss_type_percentage[ss_type_index] = [case_study_statistics[ss_type_num][name][ss_type_index].item()
                                                             for name in baseline_names]
                        plt.bar(baseline_names, ss_type_percentage[ss_type_index], width=0.4, bottom=bottom,
                                label=labels[i][ss_type_index], color=color_labels[i][ss_type_index], edgecolor='grey',
                                zorder=5)
                        bottom = [x + y for x, y in zip(bottom, ss_type_percentage[ss_type_index])]

                    plt.tick_params(axis='x', length=0)
                    plt.xlabel('Name', fontsize=12)
                    plt.ylabel('ss composition', fontsize=12)
                    plt.ylim(0, 1.01)
                    plt.yticks(np.arange(0, 1.2, 0.2), [f'{i}%' for i in range(0, 120, 20)])
                    plt.xticks(rotation=90)
                    plt.grid(axis='y', alpha=0.5, ls='--')
                    plt.legend(frameon=False, bbox_to_anchor=(1.01, 1))
                    plt.tight_layout()
                    plt.savefig(os.path.join(case_outdir,
                                             f"case_study({case_study_name})__ss{str(ss_type_num)}_statistics_{'_remove_loop' if remove_loop else ''}.png"))
                    plt.cla()

            else:               # draw global accumulative histogram
                print(baseline_composition_statistics)
                # calculate percentage of each ss composition
                for ss_type_num, ss_statistics in baseline_composition_statistics.items():
                    for name, statistics in ss_statistics.items():
                        ss_percentage = torch.tensor(statistics, dtype=torch.float)
                        if not torch.sum(ss_percentage).item() < 1:
                            ss_percentage = ss_percentage / torch.sum(ss_percentage).item()
                        baseline_composition_statistics[ss_type_num][name] = ss_percentage.numpy()
                # print(baseline_composition_statistics)

                # draw global accumulative histogram
                for i, ss_type_num in enumerate(ss_type_nums):
                    baseline_names = baseline_composition_statistics["3"].keys()
                    ss_type_percentage = dict()
                    bottom = [0] * len(baseline_names)
                    for ss_type_index in range(int(ss_type_num)):
                        ss_type_percentage[ss_type_index] = [baseline_composition_statistics[ss_type_num][name][ss_type_index].item()
                                                             for name in baseline_names]
                        plt.bar(self.get_abbreviation(baseline_names), ss_type_percentage[ss_type_index], width=0.4, bottom=bottom,
                                label=labels[i][ss_type_index], color=color_labels[i][ss_type_index], edgecolor='grey',
                                zorder=5)
                        bottom = [x + y for x, y in zip(bottom, ss_type_percentage[ss_type_index])]

                    plt.tick_params(axis='x', length=0)
                    plt.xlabel('Name', fontsize=12)
                    plt.ylabel('ss composition', fontsize=12)
                    plt.ylim(0, 1.01)
                    plt.yticks(np.arange(0, 1.2, 0.2), [f'{i}%' for i in range(0, 120, 20)])
                    plt.xticks(rotation=90)
                    plt.grid(axis='y', alpha=0.5, ls='--')
                    plt.legend(frameon=False, bbox_to_anchor=(1.01, 1))
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.outdir,
                                             f"baseline_ss{str(ss_type_num)}_statistics_{'_remove_loop' if remove_loop else ''}.png"))
                    plt.cla()

    def aa_composition(self, template_name=None, design_names=None, case_study=None):
        extended_protein_letters = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
        color_labels = [
            '#d3c6b3', '#a7adba', '#b2beb5', '#c0c0c0', '#d8bfd8',
            '#e3dac9', '#f0ead6', '#c4aead', '#b5a642', '#968d8f',
            '#bcb8b1', '#a3a9aa', '#9ea7aa', '#c4c3bf', '#dcd6d0',
            '#c1b8b7', '#9d9087', '#8e9aaf', '#b0b7c6', '#c6c7a1',
            '#e5e6e1', '#bfb8a5', '#d1cec6', '#a6a6a6', '#bab3a9',
            '#d2cfc4'
        ]




        aa_stastics = {}
        case_study_name = ''
        case_outdir = os.path.join(self.outdir, 'case_study')
        os.makedirs(case_outdir, exist_ok=True)

        # 定义 log1p 转换
        def log1p_transform(x):
            return np.log1p(100 * np.array(x)).tolist()  # 计算 log(1 + x)

        def set_radial_ticks(ax, max_value, ticks_num=5):
            y_ticks = np.linspace(0, log1p_transform(max_value), ticks_num).tolist()
            y_tick_labels = [f"{0.01 * np.expm1(y):.2f}" for y in y_ticks]  # 逆转换为原始值
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels, color="grey", size=7)
            ax.set_ylim(0, log1p_transform(max_value))

        for baseline in self.Baselines.values():
            in_dir = os.path.join(baseline.seq_dir, baseline.model_name)
            aa_stastics[baseline.model_name] = [0]*len(extended_protein_letters)

            proteins = sorted(os.listdir(in_dir))
            bar = tqdm(proteins)

            if template_name is not None:
                pass
            elif case_study is not None:
                for (baseline_abbre, design_name) in case_study:
                    if baseline_abbre == self.get_abbreviation(baseline.model_name):
                        ret = re.search('(\w+)_', design_name)
                        template = ret.group(1)
                        # Note: here consider each baseline only have one template
                        if case_study_name == '':
                            case_study_name = template

                        if 'WT' not in aa_stastics.keys():  # get WT statistics
                            WT_path = os.path.join(self.args.data_root, baseline.benchmark_config.test_set,
                                                   "aa_feature", f"{template}.pt")
                            WT_aa = torch.load(WT_path)
                            aa_stastics['WT'] = MetricsCalculation.aa_statistics(WT_aa["aa_seq"])

                        # Note: here consider each baseline only have one design
                        design_path = os.path.join(baseline.ss_dir, baseline.model_name, f"{template}.pt")
                        design_aa = torch.load(design_path)[design_name]
                        aa_stastics[baseline.model_name] = MetricsCalculation.aa_statistics(design_aa["aa_seq"])

            else:   # draw global figure
                for p in bar:
                    sequence = str(getattr(SeqIO.read(os.path.join(in_dir, p), 'fasta'), "seq"))
                    if re.search(r"original.fasta$", p):
                        aa_stastics['WT'] = MetricsCalculation.aa_statistics(sequence)
                    else:
                        aa_stastics[baseline.model_name] = MetricsCalculation.aa_statistics(sequence)


        # draw figure
        if template_name is not None:
            pass
        elif case_study is not None:
            # calculate percentage of each aa composition
            for name, statistics in aa_stastics.items():
                ss_percentage = torch.tensor(statistics, dtype=torch.float)
                if not torch.sum(ss_percentage).item() < 1:
                    ss_percentage = ss_percentage / torch.sum(ss_percentage).item()
                aa_stastics[name] = ss_percentage.numpy()

            baseline_names = aa_stastics.keys()

            # draw radar chart
            categories = list(extended_protein_letters)
            num_vars = len(categories)
            angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
            angles += angles[:1]

            plt.figure()
            fig_all, ax_all = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

            max_value = max(max(values) for values in aa_stastics.values())

            for baseline_index, name in enumerate(baseline_names):
                values = aa_stastics[name].tolist()
                log_values = log1p_transform(values)  # 转换为 log1p 坐标
                log_values += log_values[:1]

                # for all baselines
                ax_all.plot(angles, log_values, linewidth=2, linestyle='solid', label=self.get_abbreviation(name))
                ax_all.fill(angles, log_values, color_labels[baseline_index], alpha=0.1)

                # for separate baseline
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
                ax.plot(angles, log_values, linewidth=2, linestyle='solid', label=name)
                ax.fill(angles, log_values, color_labels[baseline_index], alpha=0.1)

                # 添加类别标签
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)

                # 设置径向刻度标签
                set_radial_ticks(ax, max_value)

                # # 添加标题
                # ax.set_title(f"radar_chart - {self.get_abbreviation(name)}", size=20, color=color_labels[baseline_index], y=1.1)

                # 保存单独的图像
                plt.savefig(os.path.join(case_outdir, f"radar_chart({case_study_name})_{self.get_abbreviation(name)}.png"))
                plt.close(fig)

            # 设置公用雷达图的类别标签和径向刻度标签
            ax_all.set_xticks(angles[:-1])
            ax_all.set_xticklabels(categories)
            set_radial_ticks(ax_all, max_value)

            # 添加公用雷达图的标题和图例
            ax_all.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

            # 保存公用雷达图
            plt.savefig(os.path.join(case_outdir, f"radar_chart({case_study_name})_all.png"))

        else:  # draw global figure
            # calculate percentage of each aa composition
            for name, statistics in aa_stastics.items():
                ss_percentage = torch.tensor(statistics, dtype=torch.float)
                if not torch.sum(ss_percentage).item() < 1:
                    ss_percentage = ss_percentage / torch.sum(ss_percentage).item()
                aa_stastics[name] = ss_percentage.numpy()

            baseline_names = aa_stastics.keys()

            # draw accumulative histogram
            aa_type_percentage = dict()
            bottom = [0] * len(baseline_names)
            for aa_type_index in range(len(extended_protein_letters)):
                aa_type_percentage[aa_type_index] = [
                    aa_stastics[name][aa_type_index].item()
                    for name in baseline_names]
                plt.bar(self.get_abbreviation(baseline_names), aa_type_percentage[aa_type_index], width=0.4, bottom=bottom,
                        label=extended_protein_letters[aa_type_index], color=color_labels[aa_type_index], edgecolor='grey',
                        zorder=5)
                bottom = [x + y for x, y in zip(bottom, aa_type_percentage[aa_type_index])]
            plt.tick_params(axis='x', length=0)
            plt.xlabel('Name', fontsize=12)
            plt.xticks(rotation=90)
            plt.ylabel('ss composition', fontsize=12)
            plt.ylim(0, 1.01)
            plt.yticks(np.arange(0, 1.2, 0.2), [f'{i}%' for i in range(0, 120, 20)])
            plt.grid(axis='y', alpha=0.5, ls='--')
            plt.legend(frameon=False, bbox_to_anchor=(1.01, 1))
            plt.tight_layout()
            plt.savefig(os.path.join(self.outdir,
                                     f"baseline_aa_statistics.png"))
            plt.cla()

            # draw radar chart
            categories = list(extended_protein_letters)
            num_vars = len(categories)
            angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
            angles += angles[:1]

            plt.figure()
            fig_all, ax_all = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

            max_value = max(max(values) for values in aa_stastics.values())

            for baseline_index, name in enumerate(baseline_names):
                values = aa_stastics[name].tolist()
                log_values = log1p_transform(values)  # 转换为 log1p 坐标
                log_values += log_values[:1]

                # for all baselines
                ax_all.plot(angles, log_values, linewidth=2, linestyle='solid', label=self.get_abbreviation(name))
                ax_all.fill(angles, log_values, color_labels[baseline_index], alpha=0.1)

                # for separate baseline
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
                ax.plot(angles, log_values, linewidth=2, linestyle='solid', label=name)
                ax.fill(angles, log_values, color_labels[baseline_index], alpha=0.1)

                # 添加类别标签
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)

                # 设置径向刻度标签
                set_radial_ticks(ax, max_value)

                # # 添加标题
                # ax.set_title(f"radar_chart - {self.get_abbreviation(name)}", size=20, color=color_labels[baseline_index], y=1.1)

                # 保存单独的图像
                baseline_outdir = os.path.join(self.outdir, name)
                os.makedirs(baseline_outdir, exist_ok=True)
                plt.savefig(os.path.join(baseline_outdir, f"radar_chart_{self.get_abbreviation(name)}.png"))
                plt.close(fig)

            # 设置公用雷达图的类别标签和径向刻度标签
            ax_all.set_xticks(angles[:-1])
            ax_all.set_xticklabels(categories)
            set_radial_ticks(ax_all, max_value)

            # 添加公用雷达图的标题和图例
            ax_all.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

            # 保存公用雷达图
            plt.savefig(os.path.join(self.outdir, "radar_chart_all.png"))

    def multi_design(analysis_func):
        """
        (This is a decorator)
        enable function in form of f(baseline_name, design_name) to input list design_name
        """

        def wrapper(self, baseline_name, design_name):
            result = None
            if type(design_name) == str:
                result = analysis_func(self, baseline_name, design_name)
            elif type(design_name) == list:
                for design in design_name:
                    result = analysis_func(self, baseline_name, design)
            else:
                raise TypeError(f"{design_name}")
            return result

        return wrapper

    
    # def alignment(self, baseline_name, design_name):
    #     """
    #     Alignment among the WT and design aa/ss seq
    #     :param design_name:
    #     :return:
    #     """
    #     if type(design_name) == str:
    #         self.alignment_(baseline_name, design_name)
    #     elif type(design_name) == list:
    #         for design in design_name:
    #             self.alignment_(baseline_name, design)
    #     else:
    #         raise TypeError(f"{design_name}")

    @multi_design
    def alignment(self, baseline_name, design_name):
        """
        Alignment among the WT and design aa/ss seq
        :param design_name:
        :return:
        """
        print(f"alignment: {design_name} ...")
        baseline = self.Baselines[baseline_name]

        ret = re.search('(\w+)_', design_name)
        template_name = ret.group(1)
        design_path = os.path.join(baseline.ss_dir, baseline.model_name, f"{template_name}.pt")
        WT_path = os.path.join(self.args.data_root, baseline.benchmark_config.test_set, "aa_feature", f"{template_name}.pt")
        design_ss = torch.load(design_path)[design_name]
        WT_ss = torch.load(WT_path)

        # output path
        baseline_outdir = os.path.join(self.outdir, baseline.model_name)
        os.makedirs(baseline_outdir, exist_ok=True)
        alignment_outpath = os.path.join(baseline_outdir, f"alignments({design_name}).txt")

        def aa_ss_alignment(ss_data, fp=None):
            aa_seq = ss_data['aa_seq']
            sse3_seq = ss_data['ss3_seq']
            ss3_seq = ''
            for i, aa_sub_seq in ss_data['ss3_to_aa'].items():
                ss3_seq += sse3_seq[i]*len(aa_sub_seq)
            print(ss_data['pdb'])
            print(aa_seq)
            print(ss3_seq)
            if fp:
                fp.write(f"{ss_data['pdb']}\n"
                         f"{aa_seq}\n"
                         f"{ss3_seq}\n\n")


        # union alignment
        alignments = MetricsCalculation.ss_align(WT_ss['ss3_seq'], design_ss['ss3_seq'])
        indices = alignments[0].indices

        # defination for different token
        gap_token = '-'
        align_token = '|'
        mismatch_token = '.'
        def get_letter(seq, index):
            if index == -1:
                return gap_token
            else:
                return seq[index]

        def get_seq(ss3_to_aa_dict, index):
            if index == -1:
                return ''
            else:
                return ss3_to_aa_dict[index]

        def wrap_SS(letter, length, ss_len):
            return '['+letter*ss_len+' '*(length-ss_len-2)+']'
        def wrap_AA(seq, length):
            return '['+seq+' '*(length-len(seq)-2)+']'
        def wrap_align(letter, length):
            return letter*length

        def header(header_seq, fixed_length=14):
            return header_seq + ' ' * (fixed_length - len(header_seq))

        target_additional_seq = header(f"WT AA: ")
        target_seq = header(f"WT SS: ")
        align_seq = header('')
        query_seq = header(f"DESIGN SS: ")
        query_additional_seq = header(f"DESIGN AA: ")

        for column_index in range(indices.shape[1]):
            target_index = indices[0][column_index]
            query_index = indices[1][column_index]
            target_letter = get_letter(WT_ss['ss3_seq'], target_index)
            query_letter = get_letter(design_ss['ss3_seq'], query_index)
            if target_letter == gap_token or query_letter == gap_token:
                align_letter = gap_token
            elif query_letter == target_letter:
                align_letter = align_token
            else:
                align_letter = mismatch_token
            target_additional_sub_seq = get_seq(WT_ss['ss3_to_aa'], target_index)
            query_additional_sub_seq = get_seq(design_ss['ss3_to_aa'], query_index)
            sub_seq_len = max(len(target_additional_sub_seq), len(query_additional_sub_seq))+2

            target_additional_seq += wrap_AA(target_additional_sub_seq, sub_seq_len)
            target_seq += wrap_SS(target_letter, sub_seq_len, len(target_additional_sub_seq))
            align_seq += wrap_align(align_letter, sub_seq_len)
            query_seq += wrap_SS(query_letter, sub_seq_len, len(query_additional_sub_seq))
            query_additional_seq += wrap_AA(query_additional_sub_seq, sub_seq_len)

        content = '\n'.join([target_additional_seq, target_seq, align_seq, query_seq, query_additional_seq])
        print(content)

        with open(alignment_outpath, 'w') as f:
            aa_ss_alignment(WT_ss, f)
            aa_ss_alignment(design_ss, f)
            f.write(f"{alignments[0]}\n\n")
            f.write(content)
        print(alignments[0])

    @staticmethod
    def lighten_color(color, factor=0.5):
        """
        Lightens the given color by mixing it with white.

        Parameters:
        color (array-like): The original color as an array of RGB values in the range [0, 1].
        factor (float): The factor by which to lighten the color. Should be between 0 (no change) and 1 (white).

        Returns:
        numpy array: The lightened color.
        """
        white = np.array([1.0, 1.0, 1.0])
        return color * (1 - factor) + white * factor

    @staticmethod
    def pymol_align_matrix(design_pdb, target_pdb):
        import pymol
        from pymol import cmd

        # 启动PyMOL
        cmd.reinitialize()

        # 加载蛋白质结构
        cmd.load(design_pdb, "protein1")
        cmd.load(target_pdb, "protein2")

        # 对齐结构
        cmd.align("protein1", "protein2")

        # 获取对齐后的变换矩阵
        matrix = cmd.get_object_matrix("protein1", state=1)

        # # 应用对齐的变换矩阵到protein1的坐标
        # coords = np.array(cmd.get_coords("protein1"))
        # transformed_coords = np.dot(coords, np.array(matrix).T)
        #
        # # 保存变换后的坐标为新的PDB文件
        # with open(os.path.join(self.outdir, "protein1_aligned.pdb"), "w") as f:
        #     for i, coord in enumerate(transformed_coords):
        #         f.write(
        #             f"ATOM  {i + 1:>5}  CA  PRO A   1    {coord[0]:>7.3f} {coord[1]:>7.3f} {coord[2]:>7.3f}  1.00  0.00           C\n")
        #
        # # 保存protein2的坐标为新的PDB文件
        # cmd.save(os.path.join(outdir, "protein2.pdb"), "protein2")
        return np.array(matrix).T

    @staticmethod
    def pymol_show(path):
        import pymol
        from pymol import cmd

        pymol.finish_launching()

        if type(path) is str:
            cmd.load(path)
        elif type(path) is list:
            for path_ in path:
                cmd.load(path_)

        cmd.show('cartoon')

    @staticmethod
    def align_structures(file1, file2, output_file):
        parser = PDBParser(QUIET=True)
        structure1 = parser.get_structure('structure1', file1)
        structure2 = parser.get_structure('structure2', file2)

        # Extract the atoms to be used for alignment (e.g., C-alpha atoms)
        atoms1 = [atom for atom in structure1.get_atoms() if atom.get_id() == 'CA']
        atoms2 = [atom for atom in structure2.get_atoms() if atom.get_id() == 'CA']

        # Ensure the atom lists are of the same length
        if len(atoms1) != len(atoms2):
            print("The number of C-alpha atoms does not match between the two structures.")
            return

        # Superimpose structures
        super_imposer = Bio.PDB.Superimposer()
        super_imposer.set_atoms(atoms1, atoms2)
        super_imposer.apply(structure2.get_atoms())

        # Save the aligned structure to a new PDB file
        io = PDBIO()
        io.set_structure(structure2)
        io.save(output_file)

        print(f"Aligned structure saved to {output_file}")

    @multi_design
    def ss_graph_draw(self, baseline_name, design_name):
        # path config
        print(f"alignment: {design_name} ...")
        baseline = self.Baselines[baseline_name]
        baseline_outdir = os.path.join(self.outdir, baseline.model_name)
        os.makedirs(baseline_outdir, exist_ok=True)

        ret = re.search('(\w+)_', design_name)
        template_name = ret.group(1)
        design_pdb = os.path.join(baseline.esmfold_sructure_dir, baseline.model_name, template_name, f"{design_name}.pdb")
        target_pdb = os.path.join(self.args.data_root, baseline.benchmark_config.test_set, "pdb", f"{template_name}.pdb")

        # check if design pdb exists:
        # if not os.path.isfile(design_pdb):
        #     # use esmfold for structure prediction
        #     # TODO
        #     call_esmfold(fasta_dir=in_dir, out_dir=os.path.join(baseline.esmfold_sructure_dir, baseline.model_name, template_name), length_filter=length_filter)
        
        
        # Original color map
        color_map = dict()
        color_map[0] = np.array([176 / 255, 52 / 255, 60 / 255])
        color_map[1] = np.array([210 / 255, 170 / 255, 87 / 255])
        color_map[2] = np.array([147 / 255, 168 / 255, 178 / 255])

        # Lightened color map
        lightened_color_map = {k: self.lighten_color(v, 0.8) for k, v in color_map.items()}

        # lightened_color_for_WT
        lightened_color_for_WT = False
        if lightened_color_for_WT:
            color_maps = [color_map, lightened_color_map]
        else:
            color_maps = [color_map, color_map]
        edge_colors = ['black', 'gray']
        pdbs = [design_pdb, target_pdb]

        def graph_process(pdb_file):
            data = dataset_pipeline.pdb_pipeline(args, pdb_file)
            print(data)
            # 2. 使用edge_index创建一个图
            edge_index = data.b_edge_index.t().tolist()
            G = nx.Graph(edge_index)
            # 3. 为每个节点添加属性
            #
            # for i in range(data.b_type.shape[0]):
            #     G.nodes[i]['x'] = data.b_type[i].tolist()

            for i, node_data in enumerate(data.b_type):
                G.nodes[i]['x'] = node_data.tolist()

            for i, position in enumerate(data.b_pos):
                G.nodes[i]['pos'] = position.tolist()
            return G

        def transform_graph_positions(G, matrix):
            matrix = np.array(matrix).reshape((4, 4))  # 重新形状为 (4, 4)
            for i in G.nodes.keys():
                pos_homogeneous = np.append(np.array(G.nodes[i]['pos']), 1)  # 将坐标向量变成齐次坐标
                transformed_pos = np.dot(pos_homogeneous, matrix.T)[:3]  # 矩阵乘法并截取前三个元素
                G.nodes[i]['pos'] = transformed_pos.tolist()
            return G

        # 创建组合图和单独图
        fig_design = plt.figure(figsize=(9, 9), dpi=150)
        ax_design = fig_design.add_subplot(111, projection='3d')

        fig_target = plt.figure(figsize=(9, 9), dpi=150)
        ax_target = fig_target.add_subplot(111, projection='3d')

        def draw_graph(G, ax, edge_color, node_color_map):
            edges = G.edges
            node_labels = nx.get_node_attributes(G, 'x')
            pos = nx.get_node_attributes(G, 'pos')

            # 绘制边
            for edge in edges:
                start = pos[edge[0]]
                end = pos[edge[1]]
                xs = [start[0], end[0]]
                ys = [start[1], end[1]]
                zs = [start[2], end[2]]
                ax.plot(xs, ys, zs, color=edge_color, linewidth=0.5)

            # 绘制节点
            for node_id, position in pos.items():
                x, y, z = position
                color = node_color_map[node_labels[node_id].index(1)]
                ax.scatter(x, y, z, c=color, cmap=plt.cm.get_cmap('Set1'), s=300, linewidth=1, edgecolor=edge_color,
                           label=f'Node {node_id}')

            # 隐藏坐标轴刻度和标签
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            ax.view_init(0, 0)

        # 绘制 design_pdb 和 target_pdb 的组合图
        for graph_index, pdb in enumerate(pdbs):
            G = graph_process(pdb)
            if graph_index == 0:  # design_pdb
                matrix = self.pymol_align_matrix(design_pdb, target_pdb)
                G = transform_graph_positions(G, matrix)
                draw_graph(G, ax_design, edge_colors[graph_index], color_maps[graph_index])
            else:
                draw_graph(G, ax_target, edge_colors[graph_index], color_maps[graph_index])

        fig_design.savefig(os.path.join(baseline_outdir, f"{design_name}_design_graph.png"))
        fig_target.savefig(os.path.join(baseline_outdir, f"{design_name}_target_graph.png"))

        plt.close(fig_design)
        plt.close(fig_target)

        # # abandoned code
        # # draw design & target pdb
        # for graph_index, pdb in enumerate(pdbs):
        #     G = graph_process(pdb)
        #     if graph_index == 0:    # design_pdb
        #         matrix = self.pymol_align_matrix(design_pdb, target_pdb)
        #         G = transform_graph_positions(G, matrix)
        #     edges = G.edges
        #     node_labels = nx.get_node_attributes(G, 'x')
        #     # 获取节点坐标和颜色信息
        #     pos = nx.get_node_attributes(G, 'pos')
        #     # 绘制边
        #     for edge in edges:
        #         start = pos[edge[0]]
        #         end = pos[edge[1]]
        #         xs = [start[0], end[0]]
        #         ys = [start[1], end[1]]
        #         zs = [start[2], end[2]]
        #         ax.plot(xs, ys, zs, color=edge_colors[graph_index], linewidth=0.5)
        #
        #     # 绘制节点
        #     for node_id, position in pos.items():
        #         x, y, z = position
        #         color = color_maps[graph_index][node_labels[node_id].index(1)]
        #         ax.scatter(x, y, z, c=color, cmap=plt.cm.get_cmap('Set1'), s=300, linewidth=1, edgecolor=edge_colors[graph_index],
        #                    label=f'Node {node_id}')  # ,alpha=np.min([1/np.abs(y/10),0.999]))
        #
        # # 隐藏坐标轴刻度和标签
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # ax.set_xlabel("")
        # ax.set_ylabel("")
        # ax.set_zlabel("")
        #
        # # 隐藏灰色方格背景和坐标轴线条，移除坐标轴边框
        # ax.grid(False)
        # ax.xaxis.pane.fill = False
        # ax.yaxis.pane.fill = False
        # ax.zaxis.pane.fill = False
        # ax.xaxis.pane.set_edgecolor('none')
        # ax.yaxis.pane.set_edgecolor('none')
        # ax.zaxis.pane.set_edgecolor('none')
        #
        # ax.view_init(0, 0)
        # plt.savefig(os.path.join(baseline_outdir, f"{design_name}_ss_graph.png"))






if __name__ == '__main__':
    # settings
    args = create_parser()
    print(args)
    RFdiffusion = RFdiffusionConfig()
    ESM3 = ESM3Config()
    ProtDiffS40 = ProtDiffConfig(
        name='protdiff_2nd_S40',
        diff_ckpt='results/diffusion/weight/diffusion_CATH43S40_SPLIT.pt',
        decoder_ckpt='./results/decoder/ckpt/decoder_ESM2_AFDB_AttentionPooling_LR_1E4_DROPOUT_01.pt',
        encoder_type='AttentionPooling',
        n_layers='4',
        hdim='640',
        embedding_dim='1280',
        diff_dp='0.0',
        noise_schedule='sqrt'
    )
    VanillaDecoder_random = VanillaDecoderConfig(
        name='VanillaDecoder_random',
        decoder_ckpt='./results/decoder/ckpt/decoder_ESM2_AFDB_AttentionPooling_LR_1E4_DROPOUT_01.pt',
        encoder_type='AttentionPooling',
        type="random_embedding"
    )
    VanillaDecoder_encoder_out = VanillaDecoderConfig(
        name='VanillaDecoder_encoder_out',
        decoder_ckpt='./results/decoder/ckpt/decoder_ESM2_AFDB_AttentionPooling_LR_1E4_DROPOUT_01.pt',
        encoder_type='AttentionPooling',
        type="encoder_embedding"
    )
    # if torch.cuda.is_available():
    #     ESMIF = ESMIFConfig(
    #         name='esm_if1_gvp4_t16_142M_UR50',
    #         model=esm.pretrained.esm_if1_gvp4_t16_142M_UR50()[0].eval().cuda()
    #     )
    #     esm_2_model = EsmForMaskedLM.from_pretrained(args.ESM2_dir)
    #     esm_2_model.cuda()
    #     esm_2_model.eval()
    # else:
    #     ESMIF = ESMIFConfig(
    #         name='esm_if1_gvp4_t16_142M_UR50',
    #         model=esm.pretrained.esm_if1_gvp4_t16_142M_UR50()[0].eval()
    #     )
    #     esm_2_model = EsmForMaskedLM.from_pretrained(args.ESM2_dir).eval()
    #     esm_2_model.eval()
    # esm_2_name = "esm_2"
    # ESM2 = ESM2Config(
    #     name=esm_2_name,
    #     model=esm_2_model,
    #     tokenizer=AutoTokenizer.from_pretrained(args.ESM2_dir),
    #     mask_ratio=0)
    # prostT5 = prostT5Config()
    # proteinMPNN = ProteinMPNNConfig()
    #
    # esm_2_name = "esm_2_"+f"ratio_{str(0.6).replace('.', '')}"
    # ESM2_ratio_06 = ESM2Config(
    #     name=esm_2_name,
    #     model=esm_2_model,
    #     tokenizer=AutoTokenizer.from_pretrained(args.ESM2_dir),
    #     mask_ratio=0.6,
    #     type="random_mask"
    # )
    #
    # esm_2_name = "esm_2_"+f"ratio_{str(0.8).replace('.', '')}"
    # ESM2_ratio_08 = ESM2Config(
    #     name=esm_2_name,
    #     model=esm_2_model,
    #     tokenizer=AutoTokenizer.from_pretrained(args.ESM2_dir),
    #     mask_ratio=0.8,
    #     type="random_mask"
    # )
    #
    # esm_2_name = "esm_2_"+f"ratio_{str(0.1).replace('.', '')}"
    # ESM2_ratio_01 = ESM2Config(
    #     name=esm_2_name,
    #     model=esm_2_model,
    #     tokenizer=AutoTokenizer.from_pretrained(args.ESM2_dir),
    #     mask_ratio=0.1,
    #     type="random_mask"
    # )

    # run
    benchmark_config = BenchmarkConfig()
    benchmark_pipeline = BenchmarkPipeline(ESM3, benchmark_config=benchmark_config)
    benchmark_pipeline.run()
    exit(0)

    # ESM-2 pipeline
    # esm_2_ratios=[0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]
    # for esm_2_ratio in esm_2_ratios:
    #     esm_2_name = "esm_2_"+f"ratio_{str(esm_2_ratio).replace('.', '')}"
    #     ESM2 = ESM2Config(
    #         name=esm_2_name,
    #         model=esm_2_model,
    #         tokenizer=AutoTokenizer.from_pretrained(args.ESM2_dir),
    #         mask_ratio=esm_2_ratio
    #         type="random_mask"
    #     )
    #     benchmark_config = BenchmarkConfig(test_num='50', sample_num='200')
    #     benchmark_pipeline = BenchmarkPipeline(ESM2, benchmark_config=benchmark_config)
    #     benchmark_pipeline.run()





    # name map
    # {'protdiff_2nd_S40': "CPDiffusion-SS", "esm_if1_gvp4_t16_142M_UR50": "ESM-IF1",
    #  'prostT5': "ProstT5", "ProteinMPNN": "ProteinMPNN",
    #  "esm_2": "ESM-2(1)", "esm_2_ratio_06": "ESM-2(0.6)",
    #  "esm_2_ratio_08": "ESM-2(0.8)",
    #  "esm_2_PLM": "ESM-2(1)", 'prostT5_foldseek_PLM': "ProstT5",
    #  "ProteinMPNN_IF": "ProteinMPNN", "esm_if1_gvp4_t16_142M_UR50_IF": "ESM-IF1"}

    # 2024.5.19
    # analyze benchmark
    # model_configs = [ESM2_ratio_08]
    # benchmark_config = BenchmarkConfig()
    # model_configs = [ProtDiffS40, prostT5, ESM2, proteinMPNN, ESM2_ratio_06, ESM2_ratio_08, ESMIF]
    # model_configs = [ProtDiffS40, prostT5, proteinMPNN, ESMIF]  # for case study
    # benchmark_analysis = BenchmarkAnalysis(model_configs=model_configs, args=args, benchmark_config=benchmark_config)
    # global figure
    # benchmark_analysis.ss_composition()
    # benchmark_analysis.aa_composition()

    # benchmark_analysis.alignment('ESM-IF1', ['1b8gA01_3', '1b8gA01_7'])
    # case study list:
    # figure a)
    # 'CPDiffusion-SS': ['1sumB02_198', '2dycA00_173', '5xc5A00_41']
    case_study_A = {
                    'CPDiffusion-SS': ['1sumB02_198', '2dycA00_173', '5xc5A00_41'],
                    'ProteinMPNN': ['1sumA02_0', '2dycA00_1', '5xc5A00_8'],
                    'ProstT5': ['1sumA02_0', '2dycA00_4', '5xc5A00_3'],
                    'ESM-IF1': ['1sumA02_9', '2dycA00_9', '5xc5A00_0']}
    case_study_B = {'CPDiffusion-SS': ['1dpx_211', '1fqc_421', '1pok_46']}
    case_study_C = {'CPDiffusion-SS': ['5fhiA02_91']}


    # # case study figure
    for key, case in case_study_C.items():
        benchmark_analysis.alignment(key, case)
        benchmark_analysis.ss_graph_draw(key, case)

    # composition analysis
    # case_studies = [[('CPDiffusion-SS','1sumB02_198'), ('ProteinMPNN', '1sumA02_0'), ('ProstT5', '1sumA02_0'), ('ESM-IF1', '1sumA02_9')],
    #  [('CPDiffusion-SS', '2dycA00_173'), ('ProteinMPNN', '2dycA00_1'), ('ProstT5', '2dycA00_4'), ('ESM-IF1', '2dycA00_9')],
    #  [('CPDiffusion-SS', '5xc5A00_41'), ('ProteinMPNN', '5xc5A00_8'), ('ProstT5', '5xc5A00_3'),('ESM-IF1', '5xc5A00_0')]
    #  ]
    # for case_study in case_studies:
    #     benchmark_analysis.ss_composition(case_study=case_study)
    #     benchmark_analysis.aa_composition(case_study=case_study)

    # benchmark_analysis.pymol_show(['experiments/CATH43_S40_SPLIT_TEST/design_pipeline/esmfold_structures/esm_if1_gvp4_t16_142M_UR50/1bqnA02/1bqnA02_0.pdb',
    #                                   'data/CATH43_S40_SPLIT_TEST/pdb/1bqnA02.pdb'])




    # 2024.4.29
    # rerun baseline model: prostT5, VanillaDecoder_random, ESM2, proteinMPNN, ESM2_ratio_06, ESM2_ratio_08, ProtDiffS40
    # benchmark_pipeline = BenchmarkPipeline(model_config=prostT5)
    # benchmark_pipeline.run()
    # benchmark_pipeline = BenchmarkPipeline(model_config=prostT5)
    # benchmark_pipeline.run()

    # benchmark_config = BenchmarkConfig(test_num='50', sample_num='100')
    # benchmark_pipeline = BenchmarkPipeline(benchmark_config=benchmark_config, model_config=prostT5)
    # benchmark_pipeline.run()

    # test run
    # benchmark_pipeline = BenchmarkPipeline(model_config=ESM2_ratio_01)
    # benchmark_pipeline.draw_ss_statistics(template_name='1ez0B01')

    # # others pipeline
    # benchmark_pipeline = BenchmarkPipeline(VanillaDecoder_random, fold_continue=True) # or VanillaDecoder_encoder_out
    # benchmark_pipeline = BenchmarkPipeline(VanillaDecoder_encoder_out) # or VanillaDecoder_encoder_out
    # benchmark_pipeline.run(length_filter=300)

    # run diversity() only
    # benchmark_pipeline = BenchmarkPipeline(ESM2)
    # benchmark_pipeline.diversity()

    # run designability() only for all the models

    # benchmark_pipeline = BenchmarkPipeline(ProtDiffS40)
    # benchmark_pipeline.rerun(benchmark_pipeline.designability)
    # models = [ESM2, ESMIF, proteinMPNN, prostT5]
    # for model in models:
    #     benchmark_pipeline = BenchmarkPipeline(model)
    #     benchmark_pipeline.rerun(benchmark_pipeline.designability)
    #
    # esm_2_ratios = [0.6, 0.8]
    # for esm_2_ratio in esm_2_ratios:
    #     esm_2_name = "esm_2_"+f"ratio_{str(esm_2_ratio).replace('.', '')}"
    #     ESM2 = ESM2Config(
    #         name=esm_2_name,
    #         model=esm_2_model,
    #         tokenizer=AutoTokenizer.from_pretrained(args.ESM2_dir),
    #         mask_ratio=esm_2_ratio,
    #         type="random_mask"
    #     )
    #     benchmark_pipeline = BenchmarkPipeline(ESM2)
    #     benchmark_pipeline.rerun(benchmark_pipeline.designability)


    # case study
    # benchmark_config = BenchmarkConfig(test_set='sample', test_num='1', sample_num='500')
    # # benchmark_config = BenchmarkConfig(test_set='CATH43_S40_SPLIT_TEST', test_num='50', sample_num='200')
    # benchmark_pipeline = BenchmarkPipeline(ProtDiffS40Config, benchmark_config=benchmark_config)
    # benchmark_pipeline.design(plddt_threshold=0)

    # case study pipeline
    # dataset = 'case_selection'
    # new_dataset = True
    # if new_dataset:
    #     command = f"python dataset_pipeline.py --dataset {dataset}"
    #     ret = subprocess.run(command, shell=True, encoding="utf-8", check=True)
    # benchmark_config = BenchmarkConfig(test_set=dataset, test_num='3', sample_num='10')
    # models = [ESM2, proteinMPNN, ESMIF, prostT5]
    # for model in tqdm(models, desc="models to run"):
    #     benchmark_pipeline = BenchmarkPipeline(model, benchmark_config=benchmark_config)
    #     benchmark_pipeline.design(plddt_threshold=0)


    # add std
    # models = [ProtDiffS40Config]
    # for model in models:
    #     benchmark_pipeline = BenchmarkPipeline(model)
    #     benchmark_pipeline.add_std()
    # esm_2_ratios=[0.3, 0.4, 0.6, 0.8]
    # esm_2_ratios.reverse()
    # for esm_2_ratio in esm_2_ratios:
    #     esm_2_name = "esm_2_"+f"ratio_{str(esm_2_ratio).replace('.', '')}"
    #     ESM2 = ESM2Config(
    #         name=esm_2_name,
    #         model=esm_2_model,
    #         tokenizer=AutoTokenizer.from_pretrained(args.ESM2_dir),
    #         mask_ratio=esm_2_ratio,
    #         type="random_mask"
    #     )
    #     benchmark_config = BenchmarkConfig(test_num='50', sample_num='40')
    #     benchmark_pipeline = BenchmarkPipeline(ESM2, benchmark_config=benchmark_config)
    #     benchmark_pipeline.add_std()




    # command = 'python train_encoder_decoder.py --sample --sample_nums 10 --sample_out_split --dataset sample --diff_ckpt results/diffusion/weight/20240312/diffusion_CATH43S20.pt --decoder_ckpt ./results/decoder/ckpt/20240227/decoder_ESM2_AFDB_AttentionPooling.pt --sample_out_dir benchmark/seqs/protdiff-2nd --encoder_type AttentionPooling'
    # ret = subprocess.run(command,shell=True,encoding="utf-8",timeout=1000)



