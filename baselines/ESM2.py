import torch
import esm
import re
import random
from transformers import EsmModel, EsmConfig, AutoTokenizer, EsmForMaskedLM, EsmTokenizer

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

def ESM2_generation(seq, sample_num, ratio):
    pass


if __name__ == "__main__":
    ESM2_dir = 'models/esm2_t33_650M_UR50D'
    # Load ESM-2 model
    tokenizer = AutoTokenizer.from_pretrained(ESM2_dir)
    model = EsmForMaskedLM.from_pretrained(ESM2_dir)
    model.cuda()
    model.eval()


    ori_seq = 'MKNIPSLADYPEPTHWVRFGLQNLGDADEIARLRALFDPFLARHPELELAVGDGCLTLYGPADDAGLRARAEALAAAVAAAGLPPADGAGGALA'
    seq = random_mask(ori_seq, ratio=0.2)
    print(ori_seq)
    print(seq)
    inputs = tokenizer(seq, return_tensors="pt")
    print(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))
    with torch.no_grad():
        input_ids = inputs["input_ids"].cuda()
        print(input_ids.shape)
        outputs = model(input_ids)
        logits = outputs['logits']
        print(logits.shape)
        prediction = logits.argmax(dim=-1, keepdim=False)
        print(re.sub(r"\s+", "", tokenizer.decode(prediction[0], skip_special_tokens=True)))

    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    # batch_converter = alphabet.get_batch_converter()
    # model.eval()  # disables dropout for deterministic results
    #
    # # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    # data = [
    #     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    #     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    #     ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    #     ("protein3",  "K A <mask> I S Q"),
    # ]
    # batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    #
    # # Extract per-residue representations (on CPU)
    # with torch.no_grad():
    #     results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    # token_representations = results["representations"][33]
    #
    # # Generate per-sequence representations via averaging
    # # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    # sequence_representations = []
    # for i, tokens_len in enumerate(batch_lens):
    #     sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    #
    # # Look at the unsupervised self-attention map contact predictions
    # import matplotlib.pyplot as plt
    # for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
    #     plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    #     plt.title(seq)
    #     plt.show()