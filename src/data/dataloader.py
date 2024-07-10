import os
import torch


def collate_fn(batch):
    # space-saving strategy: Padding embeddings mask every time a batch is selected

    batch_graph_list = []
    batch_aa_seq = []
    batch_seq_name = []
    badfile_flag = False
    for file_name in batch:
        graph = torch.load(os.path.join(args.diff_dir, file_name))
        deco = torch.load(os.path.join(args.deco_dir, file_name))
        raw_embedding = graph.raw_embedding
        ss_len = len(raw_embedding.keys())
        with torch.no_grad():
            # repair graph.input_mask with wrong max_len_per_ss
            if not graph.input_mask.shape[-1] == args.max_len_per_ss:
                print('repair input mask with wrong max_len_per_ss')
                input_mask_list = []
                for j in range(ss_len):
                    if raw_embedding[j].shape[0] > args.max_len_per_ss: # if file unrepairable
                        # skip this file
                        badfile_flag = True
                        break
                    input_mask_list.append(torch.cat(
                        [torch.ones(1, raw_embedding[j].shape[0]), torch.zeros(1, args.max_len_per_ss - raw_embedding[j].shape[0])],
                        dim=-1))
                else:
                    input_mask = torch.cat(input_mask_list, dim=0)
                    graph.input_mask = input_mask
                    torch.save(graph, os.path.join(args.diff_dir, file_name))
            if badfile_flag:
                badfile_flag=False
                continue
            # transform embeddings from dict to tensor
            embedding_list = []
            for j in range(ss_len):
                padding = torch.zeros(args.max_len_per_ss - raw_embedding[j].shape[0], raw_embedding[j].shape[1]).to(raw_embedding[j].device)
                paddded_embedding = torch.cat([raw_embedding[j].detach(), padding], dim=0)      # [max_len_per_ss, 1280]
                paddded_embedding = torch.unsqueeze(paddded_embedding, dim=0)   # [1, max_len_per_ss, 1280]
                embedding_list.append(paddded_embedding)
            embedding = torch.cat(embedding_list, dim=0)

        batch_graph_list.append(Data(
            x=embedding,  # [ss_len, args.max_len_per_ss, 1280]
            b_type=graph.b_type,  # [ss_len,3] one-hot
            b_pos=graph.b_pos,  # [ss_len,3]
            input_mask=graph.input_mask # [ss_len, args.max_len_per_ss]
        ))
        batch_aa_seq.append(deco["aa_seq"])
        batch_seq_name.append(deco['pdb'])
    batch_graph = Batch.from_data_list(batch_graph_list).to(device)
    batch_aa = tokenizer(batch_aa_seq, padding=True, truncation=True, return_tensors="pt")
    labels = batch_aa["input_ids"].clone()
    batch_aa['aa_seq'] = batch_aa_seq
    batch_aa['pdb'] = batch_seq_name
    labels[labels == tokenizer.pad_token_id] = -100
    batch_aa["labels"] = labels
    for k, v in batch_aa.items():
        if torch.is_tensor(v):
            batch_aa[k] = v.to(device)

    return batch_graph, batch_aa