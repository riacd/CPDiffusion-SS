import numpy as np
from pathlib import Path
import torch
import os
import esm
import esm.inverse_folding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_seq_singlechain(model, pdb_file):
    coords, native_seq = esm.inverse_folding.util.load_coords(pdb_file, "A")
    print('Native sequence loaded from structure file:')
    print(native_seq)

    sampled_seq = model.sample(coords, temperature=1, device=device)
    recovery = np.mean([(a==b) for a, b in zip(native_seq, sampled_seq)])
    print(f'Sampled sequence rr={recovery}:')
    print(sampled_seq)
    return recovery


if __name__ == '__main__':
    pdbs = os.listdir("data/CATH43_S40_TEST/pdb")[:3]
    rrs = []
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    model = model.cuda()
    for pdb in pdbs:
        pdb_path = f"data/CATH43_S40_TEST/pdb/{pdb}"
        rrs.append(sample_seq_singlechain(model, pdb_path))