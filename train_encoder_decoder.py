import argparse
import warnings
import torch
import os
import sys
import wandb
import random
import math
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
# from torch_scatter import scatter_mean
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from transformers import logging
from accelerate import Accelerator
from time import strftime, localtime
from torch_geometric.nn import DataParallel
import matplotlib.pyplot as plt
import numpy as np
import shutil


# custom module
from src.decoder.tokenizer import Tokenizer
from src.decoder.diff_decoder import Decoder
from src.module.egnn.model import EGNN_NET2
from src.diffusion.model_diffusion import GaussianDiffusion, get_named_beta_schedule, mean_flat
from src.data.data_utils import Cath
from src.args import create_parser
from src.embedding_pooling.embedding_pooling import Attention1dPooling, MeanPooling
from src.encoder_decoder.encoder_decoder import Encoder_Decoder


# set path
current_dir=os.getcwd()
sys.path.append(current_dir)
#ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
args = create_parser()
max_len_per_ss = args.max_len_per_ss


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def print_param_num(model):
    total = sum([param.numel() for param in model.parameters() if param.requires_grad])
    num_M = total/1e6
    if num_M >= 1000:
        print("Number of parameter: %.2fB" % (num_M/1e3))
    else:
        print("Number of parameter: %.2fM" % (num_M))






class Trainer:
    def __init__(self, args, encoder_decoder, accelerator,
                 train_loader, val_loader, optimizer, diffusion=None) -> None:
        self.args = args
        self.encoder_decoder = encoder_decoder
        self.accelerator = accelerator
        self.train_loader, self.val_loader = train_loader, val_loader
        self.optimizer = optimizer
        self.diffusion = diffusion

    
    def train(self, patience=10000):
        best_loss = 10
        train_loss_history, val_loss_history = [], []
        path = os.path.join(self.args.ckpt_dir, self.args.model_name)
        for epoch in range(self.args.max_train_epochs):
            print(f"---------- Epoch {epoch} ----------")
            self.encoder_decoder.train()
            train_loss = self.run_epoch(self.train_loader, epoch, "train")
            train_loss_history.append(train_loss)
            print(f'EPOCH {epoch} TRAIN loss: {train_loss:.4f}')
            
            self.encoder_decoder.eval()
            with torch.no_grad():
                val_loss = self.run_epoch(self.val_loader, epoch)
                val_loss_history.append(val_loss)
                if args.wandb:
                    wandb.log({"valid/val_loss": val_loss, "valid/epoch": epoch})
            print(f'EPOCH {epoch} VAL loss: {val_loss:.4f}')
            
            if val_loss < best_loss:
                best_loss = val_loss
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(self.encoder_decoder)
                accelerator.save(unwrapped_model.state_dict(), path)
                # torch.save(self.decoder.state_dict(), path)
                print(f'>>> BEST at epoch {epoch}, val_loss: {val_loss:.4f}')
                print(f'>>> Save model to {path}')
            
            if val_loss_history.index(min(val_loss_history)) < epoch - patience:
                print(f">>> Early stopping at epoch {epoch}")
                break

            torch.cuda.empty_cache()

    def run_epoch(self, dataloader, epoch, stage=None):
        total_loss = 0
        iter_num = len(dataloader)
        global_steps = epoch * len(dataloader)
        epoch_iterator = tqdm(dataloader)
        loss_fn = nn.CrossEntropyLoss()
        for batch in epoch_iterator:
            graph_batch, aa_batch = batch
            
            # transform graph_batch to diff_out_batch
            # [node_num, feature_size] -> [batch_size, max_node_num, feature_size]
            # if self.encoder is None:
            #     graph_embedding = graph_batch.x
            # else:
            #     graph_embedding = self.encoder.get_embedding(graph_batch)


            logits, extra = self.encoder_decoder(aa_batch['input_ids'], graph_batch)
            # shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = aa_batch['labels'][:, 1:].contiguous()

            # test:
            # sampled_seqs = []
            # logits_ = (shift_logits / 0.0001).contiguous()  # Take the last step
            # probs = F.softmax(logits_, dim=-1)
            # for batch_i in range(probs.shape[0]):
            #     sampled_tokens = torch.multinomial(probs[batch_i], 1)
            #     print('sampled tokens shape: ', sampled_tokens.shape)
            #     sampled_tokens = sampled_tokens.view(-1)
            #     batch_sampled_seqs = "".join([self.decoder.tokenizer.id_to_token(int(a)) for a in list(sampled_tokens)])
            #     sampled_seqs.append(batch_sampled_seqs)
            # print('run_epoch, one-step sampled_seq', sampled_seqs)

            
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels
            loss = loss_fn(shift_logits, shift_labels)
            total_loss += loss.item()
            global_steps += 1
            
            if stage == "train":
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
            epoch_iterator.set_postfix(loss=loss.item(), ppl=math.exp(loss.item()))
            if args.wandb:
                wandb.log({"train/train_loss": loss.item(), "train/epoch": epoch}, step=global_steps)
    
        epoch_loss = total_loss / iter_num
        return epoch_loss


    def sample(self, dataloader, temperature=1.0, max_len=1024, sample_nums=1, min_len_filter=0, max_len_filter=100):
        """
        sample sequences from decoder
        
        Args:
            temperature (float, optional): modify logits. Defaults to 1.0.
            max_len (int, optional): max generation lens. Defaults to 1024.

        Returns:
            seqs: list of sampled sequences
        """
        self.encoder_decoder.eval()
        epoch_iterator = tqdm(dataloader, desc="sample epoch")
        generation_data = dict()


        with torch.no_grad():
                max_sample_iters = tqdm(range(4*sample_nums), desc='maximum sample numbers')
                for sample_iter in max_sample_iters:
                    slowest_generation = sample_nums
                    for batch in epoch_iterator:
                        graph_batch, aa_batch = batch
                        graph_batch = Batch.to_data_list(graph_batch)

                        # take out data in graph_batch and aa_batch which has generated enough samples
                        remove_index = list()
                        aa_batch['pdb'] = list()
                        aa_batch['aa_seq'] = list()
                        for index in range(len(graph_batch)):
                            seq_name = graph_batch[index].pdb
                            aa_batch['pdb'].append(seq_name)
                            aa_batch['aa_seq'].append(graph_batch[index].aa_seq)
                            if seq_name in generation_data.keys():
                                generated_num = len(generation_data[seq_name]['generated'])
                                if slowest_generation > generated_num:
                                    slowest_generation = generated_num
                                if generated_num >= sample_nums:
                                    remove_index.append(index)
                            else:
                                slowest_generation = 0

                        # remove data in graph_batch and aa_batch according to remove_index
                        remove_index.reverse()
                        for index in remove_index:
                            graph_batch.pop(index)
                            aa_batch['pdb'].pop(index)
                            aa_batch['aa_seq'].pop(index)
                        epoch_iterator.set_postfix_str(f'dynamic batch size: {len(graph_batch)}')
                        if len(graph_batch) == 0:
                            continue
                        graph_batch = Batch.from_data_list(graph_batch)

                        # diffusion model and encoder_decoder model input different types of graph_batch (diffrence in shape of graph_batch.x)
                        # Run encoder only once
                        graph_embedding = self.encoder_decoder.get_embeddings(graph_batch=graph_batch)
                        graph_batch.x = graph_embedding # change graph_batch.x shape for diffusion
                        if self.args.random_embedding:
                            graph_embedding = torch.randn_like(graph_embedding).cuda()
                        else:
                            if self.diffusion:
                                # use Diffusion to replace encoder
                                graph_embedding1 = self.diffusion.get_embedding(graph_batch, args=self.args, denoised_fn=None)
                                mse_loss = mean_flat((graph_embedding1 - graph_embedding) ** 2)
                                mse_loss = mse_loss.mean()
                                graph_embedding = graph_embedding1
                            # else use encoder output as decoder input

                        if self.args.decoder_add_2nd_label:
                            graph_embedding = torch.cat([graph_embedding, graph_batch.b_type], dim=-1)

                        sampled_tokens = self.encoder_decoder.autoregressive_generation(graph_embedding, graph_batch, temperature, max_len)
                        for i, seq_id in enumerate(sampled_tokens):
                            seq = ''
                            for token_id in seq_id:
                                token = self.encoder_decoder.decoder.tokenizer.id_to_token(int(token_id))
                                if token != '<cls>':
                                    if token == '<eos>':
                                        break
                                    else:
                                        seq = seq + token
                            seq_name = aa_batch['pdb'][i]
                            if seq_name not in generation_data.keys():
                                generation_data[seq_name] = dict()
                                generation_data[seq_name]['generated'] = list()
                                generation_data[seq_name]['original'] = aa_batch['aa_seq'][i]
                            # # apply length filter
                            len_ratio = len(seq)/len(generation_data[seq_name]['original'])
                            if min_len_filter < len_ratio < max_len_filter:
                                generation_data[seq_name]['generated'].append(seq)
                    max_sample_iters.set_postfix_str(f'early stop progress: {slowest_generation}/{sample_nums}')
                    if self.diffusion:
                        max_sample_iters.set_postfix_str(f'MSE: {mse_loss.item()}')
                    if slowest_generation == sample_nums:
                        print(f'generation finished in {sample_iter} iterations')
                        break
                else:
                    print(f'unable to generate enough samples in {max_sample_iters} iterations')
        return generation_data

def load_diffusion(args, path):
    # accelerator = self.accelerator
    diffusion_model = GaussianDiffusion(
        model=EGNN_NET2(
            input_dim=1280, hidden_channels=args.hdim, dropout=args.diff_dp,
            n_layers=args.n_layers, update_edge=True,
            embedding_dim=args.embedding_dim, norm_feat=True,
            ss_coef=args.ss_coef
        ),
        betas=get_named_beta_schedule(
            args.noise_schedule, args.diffusion_steps
        ),
        ss_coef=args.ss_coef
    )
    data = torch.load(path)
    diffusion_model.load_state_dict(data['model'])
    if 'version' in data:
        print(f"loading from version {data['version']}")
    return diffusion_model


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
            paddded_embedding = torch.cat([raw_embedding[j], padding], dim=0)      # [max_len_per_ss, 1280]
            paddded_embedding = torch.unsqueeze(paddded_embedding, dim=0)   # [1, max_len_per_ss, 1280]
            embedding_list.append(paddded_embedding)
        embedding = torch.cat(embedding_list, dim=0)

        batch_graph_list.append(Data(
            x=embedding,  # [ss_len, args.max_len_per_ss, 1280]
            b_type=graph.b_type,  # [ss_len,3] one-hot
            b_pos=graph.b_pos,  # [ss_len,3]
            b_edge_index=graph.b_edge_index, # [2, ss_edge_num]
            b_edge_attr=graph.b_edge_attr, # [ss_edge_num, 1]
            input_mask=graph.input_mask, # [ss_len, args.max_len_per_ss]
            aa_seq=deco["aa_seq"],
            pdb=deco['pdb']
        ))
        batch_aa_seq.append(deco["aa_seq"])
        batch_seq_name.append(deco['pdb'])
    batch_graph = Batch.from_data_list(batch_graph_list)
    batch_aa = tokenizer(batch_aa_seq, padding=True, truncation=True, return_tensors="pt")
    labels = batch_aa["input_ids"].clone()
    # batch_aa['aa_seq'] = batch_aa_seq
    # batch_aa['pdb'] = batch_seq_name
    labels[labels == tokenizer.pad_token_id] = -100
    batch_aa["labels"] = labels
    for k, v in batch_aa.items():
        if torch.is_tensor(v):
            batch_aa[k] = v
    return batch_graph, batch_aa


def get_sample_dataloader():
    diff_data_list = os.listdir(args.diff_dir)
    random.seed(args.random_seed)
    random.shuffle(diff_data_list)
    if len(diff_data_list) >= args.max_data_num:
        diff_data_list = diff_data_list[:args.max_data_num]
    print(">>> sample set length: ", len(diff_data_list))

    return DataLoader(diff_data_list, batch_size=args.diff_batch_size, collate_fn=collate_fn, shuffle=False)


if __name__ == "__main__":
    # training using diffusion output
    # python3.9 train_encoder_decoder.py --wandb --wandb_run_name encoder_output_train --diff_ckpt 20231204_dataset=CATH_result_lr=0.0001_wd=1e-05_dp=0.2_hidden=128_noisy_type=cosine.pt--model_name decoder.pt
    # training with raw_embedding
    # python3.9 train_encoder_decoder.py --wandb --wandb_run_name raw_ESM2_embedding_train  --model_name decoder_ESM2.pt
    # training with raw_embedding + 2nd structure label (8type)
    # python train_encoder_decoder.py --wandb --decoder_add_2nd_label --wandb_run_name decoder_ESM2+2ndLabel  --model_name decoder_ESM2+2ndLabel.pt
    # training with raw_embedding + attention pooling (default is mean pooling)
    # python3.9 train_encoder_decoder.py --wandb --encoder_type AttentionPooling --wandb_run_name raw_ESM2_embedding_train  --model_name decoder_ESM2.pt
    # training on AFDB with raw_embedding + attention pooling (default is mean pooling)
    # python train_encoder_decoder.py --wandb --dataset AFDB --val_num 360000 --encoder_type AttentionPooling --patience 4 --wandb_run_name decoder_ESM2_AFDB_AttentionPooling_train  --model_name decoder_ESM2_AFDB_AttentionPooling.pt
    # training from ckpt on AFDB with raw_embedding + attention pooling (default is mean pooling)
    # python train_encoder_decoder.py --decoder_ckpt ./results/decoder/ckpt/20240119/decoder_ESM2_AFDB_AttentionPooling.pt --wandb --dataset AFDB --val_num 360000 --encoder_type AttentionPooling --patience 4 --wandb_run_name decoder_ESM2_AFDB_AttentionPooling_train  --model_name decoder_ESM2_AFDB_AttentionPooling.pt
    # test
    # python train_encoder_decoder.py --wandb --wandb_run_name decoder_overfitting_train --max_data_num 4 --val_num 1 --dataset sample --model_name overfitting_decoder.pt --batch_size 3 --patience 10000 --max_train_epochs 1000
    # python train_encoder_decoder.py --dataset AFDB --val_num 360000 --encoder_type AttentionPooling --patience 4 --wandb_run_name decoder_ESM2_AFDB_AttentionPooling_train  --model_name decoder_ESM2_AFDB_AttentionPooling.pt

    args = create_parser()
    print(args)

    # sample
    if args.sample:
        # generation with diffusion + decoder
        # python train_encoder_decoder.py --sample --sample_nums 1000 --sample_out_split --dataset sample --diff_ckpt 'results/diffusion/weight/20240312/diffusion_CATH43S20.pt' --decoder_ckpt './results/decoder/ckpt/20240227/decoder_ESM2_AFDB_AttentionPooling.pt' --encoder_type AttentionPooling
        # generation with decoder
        # python train_encoder_decoder.py --sample --sample_nums 1000 --sample_out_split --dataset sample --decoder_ckpt './results/decoder/ckpt/20240227/decoder_ESM2_AFDB_AttentionPooling.pt' --encoder_type AttentionPooling
        # new
        # python train_encoder_decoder.py --sample --sample_nums 2 --sample_out_split --dataset sample --diff_ckpt results/diffusion/weight/20240312/diffusion_CATH43S40.pt --decoder_ckpt ./results/decoder/ckpt/20240227/decoder_ESM2_AFDB_AttentionPooling.pt --sample_out_dir benchmark/seqs/protdiff-2nd --encoder_type AttentionPooling
        # python train_encoder_decoder.py --sample --sample_nums 2 --sample_out_split --dataset sample --decoder_ckpt ./results/decoder/ckpt/20240227/decoder_ESM2_AFDB_AttentionPooling.pt --sample_out_dir benchmark/seqs/protdiff-2nd --encoder_type AttentionPooling
        # python train_encoder_decoder.py --sample --sample_nums 100 --max_data_num 50 --sample_out_split --dataset CATH43_S40_TEST --diff_ckpt results/diffusion/weight/20240312/diffusion_CATH43S40.pt --decoder_ckpt ./results/decoder/ckpt/20240227/decoder_ESM2_AFDB_AttentionPooling.pt --sample_out_dir benchmark/seqs/protdiff-2nd --encoder_type AttentionPooling
        assert args.decoder_ckpt, "decoder_ckpt required !!"
        print('sampling...')
        args.wandb = False
        os.environ["WANDB_DISABLED"] = "true"
        if args.diff_ckpt is None:
            diffusion = None
        else:
            diffusion = load_diffusion(args, args.diff_ckpt)
        tokenizer = Tokenizer.from_pretrained("src/decoder")
        if args.encoder_type == 'MeanPooling':
            encoder = MeanPooling()
        elif args.encoder_type == 'AttentionPooling':
            encoder = Attention1dPooling(args)
        else:
            raise Exception("unknown encoder type")
        decoder = Decoder(args, tokenizer)
        encoder_decoder = Encoder_Decoder(args, encoder=encoder, decoder=decoder)
        # decoder.load_state_dict(torch.load(args.sample_ckpt))
        encoder_decoder.load_state_dict(torch.load(os.path.normpath(args.decoder_ckpt)))
        sample_loader = get_sample_dataloader()

        # accelerator = Accelerator()
        # optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
        #
        # decoder, optimizer, sample_loaderr = accelerator.prepare(
        #     decoder, optimizer, sample_loader
        # )
        print_param_num(encoder_decoder)

        accelerator = Accelerator()

        encoder_decoder, sample_loader, diffusion = accelerator.prepare(
            encoder_decoder, sample_loader, diffusion
        )

        trainer = Trainer(
            args, encoder_decoder, None,
            None, None, None, diffusion=diffusion
        )
        # generation_data:
        # {seq_name: {'generated': [seq1, seq2, ...], 'original': original_seq}}
        generation_data = trainer.sample(sample_loader, args.temperature, args.max_len, args.sample_nums, args.min_len_filter, args.max_len_filter)





        if os.path.exists(args.sample_out_dir):
            shutil.rmtree(args.sample_out_dir)
        os.makedirs(args.sample_out_dir, exist_ok=True)
        for i, seq_name in enumerate(generation_data.keys()):
            seqs = generation_data[seq_name]['generated']
            original_seq = generation_data[seq_name]['original']
            print(seq_name, ' generated: ', seqs)
            print(seq_name, ' original: ', original_seq)
            if args.sample_out_split:
                out_path = os.path.join(args.sample_out_dir, seq_name + '_original.fasta')
                with open(out_path, 'w') as f:
                    f.write('>'+seq_name+'_original'+'\n'+original_seq)
                for j, seq in enumerate(seqs):
                    out_path = os.path.join(args.sample_out_dir, seq_name+'_'+str(j)+'.fasta')
                    with open(out_path, 'w') as f:
                        f.write('>'+seq_name+'_'+str(j)+'\n'+seq)

            else:
                out_path = os.path.join(args.sample_out_dir, 'sample.fasta')
                with open(out_path, 'a') as f:
                    if i == 0:
                        f.truncate(0)
                    f.write('>' + seq_name + '_original' + '\n' + original_seq + '\n')
                    for j, seq in enumerate(seqs):
                        f.write('>' + seq_name + '_' + str(j) + '\n' + seq + '\n')
        exit(0)

    else:
        # train
        print('training...')
        if not args.wandb:
            print('wandb disabled')
            os.environ["WANDB_DISABLED"] = "true"
        if args.ckpt_dir is None:
            current_date = strftime("%Y%m%d", localtime())
            args.ckpt_dir = os.path.join(args.ckpt_root, current_date)
        os.makedirs(args.ckpt_dir, exist_ok=True)


        def seed_everywhere(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            random.seed(seed)  # Python random module.
        seed_everywhere(args.seed)

        if args.wandb:
            if args.wandb_run_name is None:
                args.wandb_run_name = args.wandb_project
            wandb.init(
                project=args.wandb_project, name=args.wandb_run_name,
                entity=args.wandb_entity, config=vars(args)
            )


        tokenizer = Tokenizer.from_pretrained("src/decoder")

        # diffusion will not be used in training process
        # if args.diff_ckpt is None:
        #     diff = None
        # else:
        #     diff = load_diffusion(args, args.diff_ckpt)

        if args.encoder_type == 'MeanPooling':
            encoder = MeanPooling()
        elif args.encoder_type == 'AttentionPooling':
            encoder = Attention1dPooling(args)
        else:
            raise Exception("unknown encoder type")

        decoder = Decoder(args, tokenizer)
        encoder_decoder = Encoder_Decoder(args, encoder=encoder, decoder=decoder)
        if args.decoder_ckpt:
            print(f'training from decoder_ckpt: {args.decoder_ckpt}')
            encoder_decoder.load_state_dict(torch.load(args.decoder_ckpt))
        print_param_num(encoder_decoder)
        print_param_num(encoder_decoder.decoder)



        # split train, val, test
        if len(os.listdir(args.diff_dir)) > args.max_data_num:
            diff_data_list = sorted(os.listdir(args.diff_dir))[:args.max_data_num]
        else:
            diff_data_list = sorted(os.listdir(args.diff_dir))
        random.shuffle(diff_data_list)
        val_num = int(len(diff_data_list)*args.val_ratio/(1-args.test_ratio))
        train_list, val_list = diff_data_list[:-val_num], diff_data_list[-val_num:]
        print(">>> trainset: ", len(train_list))
        print(">>> valset: ", len(val_list))

        train_loader = DataLoader(train_list, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
        val_loader = DataLoader(val_list, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

        accelerator = Accelerator()
        optimizer = torch.optim.AdamW(encoder_decoder.parameters(), lr=args.lr)

        encoder_decoder, optimizer, train_loader, val_loader = accelerator.prepare(
            encoder_decoder, optimizer, train_loader, val_loader
        )

        print("---------- Start Training ----------")
        trainer = Trainer(
            args, encoder_decoder, accelerator,
            train_loader, val_loader, optimizer
        )
        trainer.train(patience=args.patience)
        if args.wandb:
            wandb.finish()