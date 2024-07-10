import os
import argparse
import torch
from tqdm import tqdm
from Bio import PDB
from Bio.SeqUtils import seq1
from transformers import EsmModel, EsmConfig, AutoTokenizer

ss_alphabet = ['H', 'E', 'C']
ss_alphabet_dic = {
    "H": "H", "G": "H", "E": "E",
    "B": "E", "I": "C", "T": "C",
    "S": "C", "L": "C", "-": "C",
    "P": "C"
}


def dssp_process(pdb_file):
    pdb_parser = PDB.PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("protein", pdb_file)
    model = structure[0]
    try:
        dssp = PDB.DSSP(model, pdb_file, dssp='mkdssp')
    except:
        dssp = PDB.DSSP(model, pdb_file)

    # extract amino acid sequence
    seq = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":
                    seq.append(residue.get_resname())
    one_letter_seq = "".join([seq1(aa) for aa in seq])

    # align dssp output
    dssp_list = [i for i in dssp]
    dssp_aligned_list = []
    i = 0
    t = 0
    while i < len(one_letter_seq):
        if t < len(dssp_list):
            dssp_res = dssp_list[t]
            if dssp_res[1] != one_letter_seq[i]:
                if dssp_res[1] == 'X':
                    t += 1
                    continue
                else:
                    dssp_res = (0, one_letter_seq[i], '-')  # dssp_res[0] not used, so assign arbitrary value
                    t -= 1
        else:
            dssp_res = (0, one_letter_seq[i], '-')
            t -= 1
        dssp_aligned_list.append(dssp_res)
        i += 1
        t += 1

    return structure, dssp, one_letter_seq, dssp_aligned_list

def generate_ss2aa(pdb_file, dssp_data=None):
    if dssp_data is None:
        dssp_data = dssp_process(pdb_file)
    structure, dssp, one_letter_seq, dssp_aligned_list = dssp_data

    # extract secondary structure sequence
    # ss8 extraction
    ss8_to_aa_len, ss8_to_aa = {}, {}
    previous_ss = None
    sec_structures = []
    j = -1
    for i, dssp_res in enumerate(dssp_aligned_list):
        if dssp_res[2] != previous_ss:
            j += 1
            sec_structures.append(dssp_res[2])
            previous_ss = dssp_res[2]
        if j not in ss8_to_aa_len.keys():
            ss8_to_aa_len[j] = 0
        ss8_to_aa_len[j] += 1
    sec_structure_str_8 = ''.join(sec_structures)
    sec_structure_str_8 = sec_structure_str_8.replace('-', 'L')
    offset = 0
    for ss_id, aa_len in ss8_to_aa_len.items():
        ss8_to_aa[ss_id] = one_letter_seq[offset:offset + aa_len]
        offset += aa_len

    # extract secondary structure sequence
    # ss3 extraction
    # TODO: second structures H & E (helix and sheet) with aa length == 1 shall be removed (whether this works requires further study)
    ss3_to_aa_len, ss3_to_aa = {}, {}
    previous_ss = None
    sec_structures = []
    j = -1
    # aa_len_per_ss = 0
    for i, dssp_res in enumerate(dssp_aligned_list):
        # aa_len_per_ss += 1
        ss_token = ss_alphabet_dic[dssp_res[2]]
        if ss_token != previous_ss:
            # if aa_len_per_ss == 2 and (previous_ss == 'H' or previous_ss == 'E'):  # remove second structure with aa length == 1
            #     sec_structures.pop()
            # else:
            #     j += 1
            #     aa_len_per_ss = 0
            j += 1
            sec_structures.append(ss_token)
            previous_ss = ss_token
        if j not in ss3_to_aa_len.keys():
            ss3_to_aa_len[j] = 0
        ss3_to_aa_len[j] += 1
    sec_structure_str_3 = ''.join(sec_structures)
    offset = 0
    for ss_id, aa_len in ss3_to_aa_len.items():
        ss3_to_aa[ss_id] = one_letter_seq[offset:offset + aa_len]
        offset += aa_len

    final_feature = {}
    final_feature["aa_seq"] = one_letter_seq
    final_feature["ss8_seq"] = sec_structure_str_8
    final_feature["ss3_seq"] = sec_structure_str_3
    final_feature["ss8_to_aa"] = ss8_to_aa
    final_feature["ss3_to_aa"] = ss3_to_aa
    final_feature["pdb"] = pdb_file.split('/')[-1].split('.')[0]

    return final_feature



def generate_embedding(args, pdb_file, plm_model, tokenizer, dssp_data=None):
    if dssp_data is None:
        dssp_data = dssp_process(pdb_file)
    structure, dssp, one_letter_seq, dssp_aligned_list = dssp_data

    inputs = tokenizer(one_letter_seq, return_tensors="pt")
    with torch.no_grad():
        input_ids = inputs["input_ids"].cuda()
        outputs = plm_model(input_ids)
        features = outputs["last_hidden_state"].squeeze(0)[1:-1, :]

    # extract amino acid coordinates
    coords = {"N": [], "CA": [], "C": [], "O": []}
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":
                    for atom_name in coords.keys():
                        if residue.has_id(atom_name):
                            atom = residue[atom_name]
                            coords[atom_name].append(atom.get_coord().tolist())
                        else:
                            coords[atom_name].append('NaN')
    # dssp_seq = [dssp_res[1] for dssp_res in dssp]
    # if len(one_letter_seq) != len(dssp):
    #     print('pdb', pdb_file)
    #     print('seqlen', len(one_letter_seq))
    #     print('seq', one_letter_seq)
    #     print('coords: ', len(coords['N']))
    #     print('dssp: ', len(dssp))
    #     print('dssp_seqlen', len(dssp_seq))
    #     print('dssp_seq', "".join(dssp_seq))


    # for each secondary structure, extract the coordinates of all amino acids
    dssp_list = [i for i in dssp]
    coord_anchors = ["N", "CA", "C", "O"]
    ss_coords_dict = {"embedding": {}, "N": {}, "CA": {}, "C": {}, "O": {}}
    final_feature = {"N": [], "CA": [], "C": [], "O": []}
    previous_ss = None
    sec_structures = []
    j = -1
    i = 0
    t = 0
    # for each amino acid, find its secondary structure
    # j: currently at the j-th second structure
    # i: currently at the i-th amino acid
    # t: currently at the t-th amino acid in dssp seq (dssp seqs might miss some amino acids, or insert aa with type X compared to original seqs)
    while i < len(one_letter_seq):
        if t < len(dssp_list):
            dssp_res = dssp_list[t]
            if dssp_res[1] != one_letter_seq[i]:
                if dssp_res[1] == 'X':
                    t += 1
                    continue
                else:
                    dssp_res = (0, one_letter_seq[i], '-') # dssp_res[0] not used, so assign arbitrary value
                    t -= 1
        else:
            dssp_res = (0, one_letter_seq[i], '-')
            t -= 1
        if args.split_with_ss8:
            ss_token = dssp_res[2]
        else:
            ss_token = ss_alphabet_dic[dssp_res[2]]
        if ss_token != previous_ss:
            j += 1
            sec_structures.append(ss_token)
            previous_ss = ss_token
        for anchor in coord_anchors:
            if j not in ss_coords_dict[anchor].keys():
                ss_coords_dict[anchor][j] = []
            if not coords[anchor][i] == 'NaN':
                ss_coords_dict[anchor][j].append(coords[anchor][i])
        if j not in ss_coords_dict["embedding"].keys():
            ss_coords_dict["embedding"][j] = features[i].unsqueeze(0)
        else:
            ss_coords_dict["embedding"][j] = torch.cat((ss_coords_dict["embedding"][j], features[i].unsqueeze(0)), 0)
        i += 1
        t += 1


    # calculate the mean coordinates of each secondary structure
    for anchor in coord_anchors:
        for k, v in ss_coords_dict[anchor].items(): # k: k-th second structure; v: list of coordinates [(x0, y0, z0), (x1, y1, z1), ...]
            ss_coords_dict[anchor][k] = [sum(e)/len(e) for e in zip(*v)] # zip(*v): [(x0, x1, x2, ...), (y0, y1, y2, ...), (z0, z1, z2, ...)]
            final_feature[anchor].append(ss_coords_dict[anchor][k])
    sec_structure_str = ''.join(sec_structures)
    if args.split_with_ss8:
        sec_structure_str = sec_structure_str.replace('-', 'L')
    final_feature["embedding"] = ss_coords_dict["embedding"]
    final_feature["ss_seq"] = sec_structure_str
    final_feature["seq"] = one_letter_seq
    final_feature["pdb"] = pdb_file.split('/')[-1].split('.')[0]
    # print(len(final_feature["embedding"]), len(final_feature["N"]), len(final_feature["CA"]), len(final_feature["C"]), len(final_feature["O"]))

    return final_feature
            

