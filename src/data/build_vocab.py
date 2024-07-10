import torch
import argparse
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP


def get_struc2ndRes(pdb_filename):
    struc_2nds_res_alphabet = ['E', 'L', 'I', 'T', 'H', 'B', 'G', 'S']
    char_to_int = dict((c, i) for i, c in enumerate(struc_2nds_res_alphabet))
    p = PDBParser()
    structure = p.get_structure('', pdb_filename)
    model = structure[0]
    dssp = DSSP(model, pdb_filename,dssp='mkdssp')
    sec_structures = [dssp_res[2] for dssp_res in dssp]
    sec_structure_str = ''.join(sec_structures)
    sec_structure_str = sec_structure_str.replace('-','L')
    integer_encoded = [char_to_int[char] for char in sec_structure_str]
    return integer_encoded

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_filename', type=str, default='1a3a.pdb')
    args = parser.parse_args()
    data = get_struc2ndRes(args.pdb_filename)
    print(data)