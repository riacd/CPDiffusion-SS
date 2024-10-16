import torch
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
import os
print(torch.__version__)
print(torch.cuda.is_available()) # 查看CUDA是否可用
print(torch.cuda.device_count()) # 查看可用的CUDA数量
print(torch.version.cuda) # 查看CUDA的版本号

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Will instruct you how to get an API key from huggingface hub, make one with "Read" permission.
# login()

# This will download the model weights and instantiate the model on your machine.
model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to(torch.device('cuda')) # or "cpu"
print("generating using:", model.tokenizers.secondary_structure.kind)

# Generate a completion for a partial Carbonic Anhydrase (2vvb)
prompt = "HHHHHEEEEEEECCCCC"
protein = ESMProtein(secondary_structure=prompt)
# Generate the sequence, then the structure. This will iteratively unmask the sequence track.
protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))
print(protein.sequence)
exit(0)
# We can show the predicted structure for the generated sequence.
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
protein.to_pdb("./generation.pdb")
# Then we can do a round trip design by inverse folding the sequence and recomputing the structure
protein.sequence = None
protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8))
protein.coordinates = None
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
protein.to_pdb("./round_tripped.pdb")