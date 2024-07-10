import argparse
import os
import re
import json
import numpy as np
import shutil
# import pymol
import Bio
from Bio import PDB
from Bio.SeqUtils import seq1
from Bio.Align import PairwiseAligner
from Bio import SeqIO
from Bio.Align import substitution_matrices
import subprocess
import os
import argparse
import pandas as pd
from tqdm import tqdm


# conda install -c schrodinger tmalign

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




def create_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--colabfold_input_dir", type=str, default=r"E:\Projects\protdiff-2nd\results\sample", help="input dir of colabfold")
    # parser.add_argument("--colabfold_output_dir", type=str, default=r"E:\Projects\protdiff-2nd\results\folding", help="output dir of colabfold")
    # parser.add_argument("--process_output_dir", type=str, default=r"E:\Projects\protdiff-2nd\results\final", help="output dir of process program")
    # parser.add_argument("--top_n_selection", type=int, default=3, help="select pdb output with top n highest avg_plddt")
    parser.add_argument("--colabfold_input_dir", type=str, default=r"/public/home/huyutong/protdiff-2nd/results/sample", help="input dir of colabfold")
    parser.add_argument("--colabfold_output_dir", type=str, default=r"/public/home/huyutong/protdiff-2nd/results/folding", help="output dir of colabfold")
    parser.add_argument("--process_output_dir", type=str, default=r"/public/home/huyutong/protdiff-2nd/results/final", help="output dir of process program")
    parser.add_argument("--top_n_selection", type=int, default=3, help="select pdb output with top n highest avg_plddt")

    args = parser.parse_args()

    os.makedirs(args.process_output_dir, exist_ok=True)
    return args

def get_avg_plddt(args, folding_name: str):
    files = os.listdir(args.colabfold_output_dir)
    for file in files:
        ret = re.match(folding_name+'_scores_rank_001\w+\.json', file)
        if ret:
            plddt_json = ret.group(0)
            break
    else:
        raise Exception("invalid folding name")
    with open(os.path.join(args.colabfold_output_dir, plddt_json), 'r') as f:
        scores_dict = json.load(f)
    plddt_score = np.array(scores_dict['plddt'])
    return plddt_score.mean().item()

def get_path(args, file_list, folding_name):
    # find pdb path for each folding
    for file in file_list:
        ret = re.match(folding_name+'_relaxed_rank_001\w+\.pdb', file)
        if ret:
            pdb_name = ret.group(0)
            break
    else:
        for file in file_list:
            ret = re.match(folding_name + '_unrelaxed_rank_001\w+\.pdb', file)
            if ret:
                pdb_name = ret.group(0)
                break
        else:
            raise Exception("cannot find corresponding pdb file")
    
    pdb = os.path.join(args.colabfold_output_dir, pdb_name)
    plddt_plot = os.path.join(args.colabfold_output_dir, folding_name+'_plddt.png')
    pae_plot = os.path.join(args.colabfold_output_dir, folding_name+'_pae.png')    
    return pdb, plddt_plot, pae_plot

def cal_seq_id(args):
    identity_dict = dict()
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    # aligner.mode = 'global'
    # aligner.match_score = 2
    # aligner.mismatch_score = 0
    fasta_files = os.listdir(args.colabfold_input_dir)
    templates = os.listdir(args.process_output_dir)
    for template in templates:
        if os.path.isdir(os.path.join(args.process_output_dir, template)):
            files = os.listdir(os.path.join(args.process_output_dir, template))
            for file in files:
                ret = re.match(f"({template}+_[0-9]+)_plddt.png", file)
                if ret:
                    gen_seq_name = ret.group(1)
                    seq_1_fasta = os.path.join(args.colabfold_input_dir, gen_seq_name+'.fasta')
                    seq_2_fasta = os.path.join(args.colabfold_input_dir, template+'_original.fasta')
                    seq_1 = SeqIO.read(seq_1_fasta, 'fasta')
                    seq_2 = SeqIO.read(seq_2_fasta, 'fasta')
                    alignments = aligner.align(str(seq_1.seq), str(seq_2.seq))
                    gaps, mismatch, identities = alignments[0].counts().gaps, alignments[0].counts().mismatches, alignments[0].counts().identities
                    identity = identities/(gaps+mismatch+identities)*100
                    print(gen_seq_name+'.fasta identity: ', identity)
                    identity_dict[gen_seq_name] = identity
    identity_dict_path = os.path.join(args.process_output_dir, 'identity.json')
    with open(identity_dict_path, 'w+') as f:
        json.dump(identity_dict, f)

def pdb_align(args):
    templates = os.listdir(args.process_output_dir)
    for template in templates:
        if os.path.isdir(os.path.join(args.process_output_dir, template)):
            files = os.listdir(os.path.join(args.process_output_dir, template))
            for file in files:
                ret = re.match(f"{template}_original\S+\.pdb", file)
                if ret:
                    reference_pdb = os.path.join(args.process_output_dir, template, ret.group(0))
            for file in files:
                ret = re.match(f"({template}_[0-9]+)\S+\.pdb", file)
                if ret:
                    predicted_pdb = os.path.join(args.process_output_dir, template, ret.group(0))
                    gen_seq_name = ret.group(1)
                    align_info = calculate_align_info(predicted_pdb, reference_pdb)
                    align_infos = {
                        "predicted_pdb": gen_seq_name,
                        "reference_pdb": template+"_original",
                        # "aligned_length": align_info["aligned_length"],
                        # "rmsd": align_info["rmsd"],
                        "seq_identity": align_info["seq_identity"],
                        "tm_score": align_info["tm_score"],
                        "tm_score_1": align_info["tm_score_1"],
                        "tm_score_2": align_info["tm_score_2"]
                    }
                    align_infos = pd.DataFrame(align_infos, index=[0])
                    align_infos.to_csv(os.path.join(args.process_output_dir, "align_info.csv"), index=False)
                    print(gen_seq_name, "identity:", align_info["seq_identity"])

def select_top_n_foldings(args):
    files = os.listdir(args.colabfold_output_dir)

    # naming rules for seqs folded are as follows:
    # 1. generated seqs(i is a number): template_name + '_' + str(i)
    # 2. original seqs: template_name + '_original'

    # find names of proteins folded with colabfold
    folding_names = []
    for file_name in files:
        ret = re.match('(\w+)_plddt', file_name)
        if ret:
            folding_names.append(ret.group(1))

    # find template names
    template_names = []
    for name in folding_names:
        ret = re.match('(\w+)_original', name)
        if ret:
            template_names.append(ret.group(1))

    # folding_dict:
    # {template_name: {
    #     'original': {
    #         'name': original_folding_name,c
    #         'avg_plddt': avg_plddt,
    #     },
    #     'generated': {
    #         'name': [fold_name1, fold_name2, ...],
    #         'avg_plddt': [(fold_name1, avg_plddt1), (fold_name2, avg_plddt2), ...],   # descending ranked with avg_plddt
    #         'all_avg_plddt': all_avg_plddt,
    #         'top_n_all_avg_plddt': top_n_all_avg_plddt,
    #         'n': args.top_n_selection
    #     }}
    # , ...}
    folding_dict = dict()
    for template_name in template_names:
        folding_dict[template_name] = dict()
        folding_dict[template_name]['original'] = dict()
        folding_dict[template_name]['generated'] = dict()
        folding_dict[template_name]['original']['name'] = template_name + '_original'
        folding_dict[template_name]['original']['avg_plddt'] = get_avg_plddt(args,
                                                                             folding_dict[template_name]['original'][
                                                                                 'name'])
        folding_dict[template_name]['generated']['name'] = list()
        folding_dict[template_name]['generated']['avg_plddt'] = list()
        for folding_name in folding_names:
            ret = re.match(template_name + '_[0-9]+', folding_name)
            if ret:
                folding_dict[template_name]['generated']['name'].append(ret.group(0))
        for i, folding_name in enumerate(folding_dict[template_name]['generated']['name']):
            folding_dict[template_name]['generated']['avg_plddt'].append(
                (folding_name, get_avg_plddt(args, folding_name)))
        folding_dict[template_name]['generated']['avg_plddt'].sort(key=lambda x: x[1], reverse=True)
        all_avg_plddt = np.array([x[1] for x in folding_dict[template_name]['generated']['avg_plddt']]).mean().item()
        top_n_all_avg_plddt = np.array([folding_dict[template_name]['generated']['avg_plddt'][i][1] for i in
                                        range(args.top_n_selection)]).mean().item()
        folding_dict[template_name]['generated']['all_avg_plddt'] = all_avg_plddt
        folding_dict[template_name]['generated']['top_n_all_avg_plddt'] = top_n_all_avg_plddt
        folding_dict[template_name]['generated']['n'] = args.top_n_selection

        print(template_name, ': ')
        print('all_avg_plddt: ', all_avg_plddt)
        print('top', args.top_n_selection, 'all_avg_plddt: ', top_n_all_avg_plddt)
        print('original_avg_plddt: ', folding_dict[template_name]['original']['avg_plddt'])

        # copy selected files to destination
        # original seq
        output_dir = os.path.join(args.process_output_dir, template_name)
        os.makedirs(output_dir, exist_ok=True)
        pdb, plddt_plot, pae_plot = get_path(args, files, folding_dict[template_name]['original']['name'])
        shutil.copy(pdb, output_dir)
        shutil.copy(plddt_plot, output_dir)
        shutil.copy(pae_plot, output_dir)
        # pymol.cmd.load(os.path.join(output_dir, pdb))
        # generated seq
        for i in range(args.top_n_selection):
            pdb_gen, plddt_plot, pae_plot = get_path(args, files,
                                                     folding_dict[template_name]['generated']['avg_plddt'][i][0])
            shutil.copy(pdb_gen, output_dir)
            shutil.copy(plddt_plot, output_dir)
            shutil.copy(pae_plot, output_dir)
            # gen = pymol.cmd.get_ob
            # result = pymol.cmd.align()

    folding_dict_output = os.path.join(args.process_output_dir, 'folding_process.json')
    with open(folding_dict_output, 'w+') as f:
        json.dump(folding_dict, f)


if __name__ == '__main__':
    args = create_parser()
    # select_top_n_foldings(args)
    # cal_seq_id(args)
    pdb_align(args)





