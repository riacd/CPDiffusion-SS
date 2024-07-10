#### main for ss diffusion ####

import os
import sys
from pathlib import Path
import random
import numpy as np
from functools import partial
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch_geometric.data import Batch,Data
from torch_geometric.loader import DataListLoader
from torch.utils.data import DataLoader
from torch_geometric.nn import DataParallel
from time import strftime, localtime
import wandb
import math

from src.data.data_utils import Cath
from src.data.cath_2nd import aa_vocab
from src.args import create_parser
from src.module.egnn.model import EGNN_NET2
from src.diffusion.model_diffusion import GaussianDiffusion, get_named_beta_schedule
from src.decoder.tokenizer import Tokenizer
from src.decoder.diff_decoder import Decoder
from src.embedding_pooling.embedding_pooling import Attention1dPooling, MeanPooling
from src.encoder_decoder.encoder_decoder import Encoder_Decoder

# set path
current_dir=os.getcwd()
sys.path.append(current_dir)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args = create_parser()


def cycle(dl):
    while True:
        for data in dl:
            yield data


def collate_fn(batch):
    # space-saving strategy: Padding embeddings mask every time a batch is selected

    batch_graph_list = []
    batch_aa_seq = []
    batch_seq_name = []
    badfile_flag = False
    for file_name in batch:
        graph = torch.load(os.path.join(args.diff_dir, file_name)).to(device)
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
                    input_mask = torch.cat(input_mask_list, dim=0).to(device)
                    graph.input_mask = input_mask
                    torch.save(graph, os.path.join(args.diff_dir, file_name))
            if badfile_flag:
                badfile_flag=False
                continue
            # transform embeddings from dict to tensor
            embedding_list = []
            for j in range(ss_len):
                padding = torch.zeros(args.max_len_per_ss - raw_embedding[j].shape[0], raw_embedding[j].shape[1]).to(device)
                paddded_embedding = torch.cat([raw_embedding[j].detach(), padding], dim=0)      # [max_len_per_ss, 1280]
                paddded_embedding = torch.unsqueeze(paddded_embedding, dim=0)   # [1, max_len_per_ss, 1280]
                embedding_list.append(paddded_embedding)
            embedding = torch.cat(embedding_list, dim=0)

        batch_graph_list.append(Data(
            x=embedding,  # [ss_len, args.max_len_per_ss, 1280]
            b_type=graph.b_type,  # [ss_len,3] one-hot
            b_pos=graph.b_pos,  # [ss_len,3]
            b_edge_index=graph.b_edge_index, # [2, ss_edge_num]
            b_edge_attr=graph.b_edge_attr, # [ss_edge_num, 1]
            input_mask=graph.input_mask # [ss_len, args.max_len_per_ss]
        ))

    batch_graph = Batch.from_data_list(batch_graph_list).to(device)

    return batch_graph


class Trianer(object):
    def __init__(
            self,
            config,
            diffusion_model,
            train_dataset,
            val_dataset,
            # test_dataset,
            *,
            train_batch_size=256,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            weight_decay=1e-2,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,  # 0.999
            adam_betas=(0.9, 0.99),
            save_and_sample_every=10000,
            num_samples=25,
            results_folder='./results/diffusion',
            resume_checkpoint = False,
    ):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.min_val = 10000

        self.ds = train_dataset
        self.model = diffusion_model.to(device)
        # dl = DataListLoader(self.ds, batch_size=train_batch_size, shuffle=True, num_workers=4)
        dl = DataLoader(self.ds, collate_fn=collate_fn, batch_size=train_batch_size, shuffle=True)
        self.dl = cycle(dl)
        print("DL OK ")

        # self.model = diffusion_model
        self.config = config
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        # dataset and dataloader
        self.val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=train_batch_size // 2, shuffle=False)
        # self.test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=train_batch_size // 2, shuffle=False,
        #                               num_workers=1)
        self.opt = AdamW(self.model.parameters(), lr=train_lr, betas=adam_betas, weight_decay=weight_decay)

        # for logging results in a folder periodically

        # if self.accelerator.is_main_process:
        # self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        Path(results_folder + '/weight/').mkdir(exist_ok=True)
        Path(results_folder + '/figure/').mkdir(exist_ok=True)
        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.current_time = strftime("%Y%m%d", localtime())
        self.save_file_name = self.config['diff_model_name']
        self.resume_checkpoint = resume_checkpoint

        # load Pooling parameters from encoder_decoder ckpt
        tokenizer = Tokenizer.from_pretrained("src/decoder")
        if args.encoder_type == 'MeanPooling':
            encoder = MeanPooling()
        elif args.encoder_type == 'AttentionPooling':
            encoder = Attention1dPooling(args)
        else:
            raise Exception("unknown encoder type")
        decoder = Decoder(args, tokenizer)
        self.encoder_decoder = Encoder_Decoder(args, encoder=encoder, decoder=decoder)
        # decoder.load_state_dict(torch.load(args.sample_ckpt))
        self.encoder_decoder.load_state_dict(torch.load(os.path.normpath(args.decoder_ckpt)))
        self.encoder_decoder.encoder.to(device)

    def save(self):
        # if not self.accelerator.is_local_main_process:
        #     return
        if hasattr(self.model, "device_ids") and len(self.model.device_ids) > 1:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        data = {
            'config': self.config,
            'step': self.step,
            'model': state_dict,
            'opt': self.opt.state_dict(),
            # 'ema': self.ema.state_dict(),
            # 'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            # 'version': __version__
        }
        saving_dir = os.path.join(str(self.results_folder), 'weight', self.current_time)
        os.makedirs(saving_dir, exist_ok=True)
        torch.save(data, str(os.path.join(saving_dir, self.save_file_name+'.pt')))

    def load(self, filename=False):
        # accelerator = self.accelerator
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if filename:
            data = torch.load(str(self.results_folder) + '/' + filename, map_location=device)
        else:
            data = torch.load(os.path.join(str(self.results_folder), 'weight', self.save_file_name + '.pt'), map_location=device)
                # str(self.results_folder / self.config[
                # 'Date'] + f"model_lr={self.config['lr']}_dp={self.config['dp']}_timestep={self.config['timesteps']}_hidden={self.config['hidden_dim']}_noisy_type={self.config['noise_schedule']}_{milestone}.pt"),
                #               map_location=device)

        # model = self.accelerator.unwrap_model(self.model)
        # clean_dict = {}
        # for key,value in data['model'].items():
        #     clean_dict[key.replace('module.','')] = value
        # model.state_dict()[key.replace('module.','')] = value
        self.model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        # self.ema.load_state_dict(data['ema'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        # if exists(self.accelerator.scaler) and exists(data['scaler']):
        #     self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):

        if self.resume_checkpoint:
            print("Loading from checkpoint...")
            self.load()

        lr_schedule = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_loss_list, ce_loss_list, mse_loss_list, recovery_list, perplexity, corr_record = [], [], [], [], [], []
        train_loss_local, val_loss_list, val_ce_loss_list, val_mse_loss_list, val_recovery_list = [], [], [], [], []
        val_accuracy_list = []
        train_ce_local, train_mse_local = [], []
        train_ce_list, train_mse_list = [], []
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:

                self.model.train()
                data = next(self.dl)
                data.x = self.encoder_decoder.get_embeddings(data)
                # data = Batch.to_data_list(data)
                # loss, Lt_loss, ce_loss, LT_loss = self.model(data)
                loss, mse_loss, ce_loss, accuracy = self.model(data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                train_loss_local.append(loss.item())
                train_ce_local.append(ce_loss.item())
                train_mse_local.append(mse_loss.item())
                if args.wandb:
                    wandb.log({"train/train_loss": loss.item(), "train/ce_loss": ce_loss.item(),"train/mse_loss": mse_loss.item()})
                pbar.set_description(f'Train total loss: {loss.item():.4f}, mse loss: {mse_loss.item():.4f}, ce loss(*ss_coef): {ce_loss.item():.4f}')

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                # if self.step > self.train_num_steps / 2 and lr_schedule:
                if self.step >= 1000 and lr_schedule:
                    for g in self.opt.param_groups:
                        g['lr'] = self.config['diff_lr'] * 0.1
                    print(f"Using new lr {g['lr']}")
                    lr_schedule = False

                # self.ema.to(device)
                # self.ema.update()
                if self.step % 100 == 0:
                    torch.cuda.empty_cache()

                if self.step % (self.save_and_sample_every * (len(self.ds) // self.batch_size)) == 1:
                    self.model.eval()

                    train_loss_list.append(torch.tensor(train_loss_local).mean().item())
                    train_mse_list.append(torch.tensor(train_mse_local).mean().item())
                    train_ce_list.append(torch.tensor(train_ce_local).mean().item())
                    train_loss_local = []
                    train_mse_local = []
                    train_ce_local = []

                    # model_emb = torch.nn.Embedding(
                    #     num_embeddings=len(aa_vocab),
                    #     embedding_dim=config['embedding_dim'],
                    #     _weight=self.ema.ema_model.model.aa_embedding.weight.clone().cpu()
                    # ).eval().requires_grad_(False)

                    with torch.no_grad():
                        sub_list = []
                        val_loss_local, val_mse_loss_local, val_ce_loss_local, accuracy_local = [], [], [], []
                        for data in self.val_loader:
                            data.x = self.encoder_decoder.get_embeddings(data)
                            # val_loss, val_Lt_loss, val_ce_loss, val_LT_loss = self.ema.ema_model(data)
                            val_loss, val_mse, val_ce, accuracy = self.model(data)
                            val_loss_local.append(val_loss.item())
                            val_mse_loss_local.append(val_mse.item())
                            val_ce_loss_local.append(val_ce.item())
                            accuracy_local.append(accuracy.item())

                            # test samples
                            if self.step % (10 * (len(self.ds) // self.batch_size)) == 1:
                                shape = [*data.x.shape]
                                sample = self.model.p_sample_loop(
                                    data, shape,
                                    denoised_fn=None,
                                    clip_denoised=args.clip_denoised,
                                    clamp_step=args.clamp_step,
                                    gap=1
                                )[-1]['sample']
                                val_sample_mse = F.mse_loss(sample, data.x)
                                sub_list.append(val_sample_mse.item())
                            # # sample = self.ema.ema_model.ddim_sample_loop(data, shape,
                            # #                                              # denoised_fn=partial(denoised_fn_round, model_emb),
                            # #                                              clip_denoised = False,
                            # #                                              gap=10)[-1] # get the last sample
                            # recovery calc takes too much time
                            # sample = self.model.p_sample_loop(data, shape,
                            #                                          # denoised_fn=partial(denoised_fn_round, model_emb),
                            #                                          clip_denoised = False,
                            #                                          clamp_step=20,
                            #                                          gap=1)[-1]['sample'] # get the last sample
                            # logits = self.ema.ema_model.model.get_logits(sample)
                            # seq_pred = logits.argmax(dim=-1)
                            # recovery = (seq_pred == data.x).float().mean()
                            # val_sample_mse = F.mse_loss(sample, data.x)
                            #
                            # ll_fullseq = F.cross_entropy(zt, data.x, reduction='mean').item()

                        val_loss_mean = torch.tensor(val_loss_local).mean().item()
                        val_mse_mean = torch.tensor(val_mse_loss_local).mean().item()
                        val_ce_mean = torch.tensor(val_ce_loss_local).mean().item()
                        if len(sub_list) > 0:
                            val_recovery_mean = torch.tensor(sub_list).mean().item()
                        val_accuracy_mean = torch.tensor(accuracy_local).mean().item()
                        val_loss_list.append(val_loss_mean)
                        val_mse_loss_list.append(val_mse_mean)
                        val_ce_loss_list.append(val_ce_mean)
                        # val_recovery_list.append(val_recovery_mean)
                        val_accuracy_list.append(val_accuracy_mean)
                        if args.wandb:
                            wandb.log({"valid/val_loss_mean": val_loss_mean})
                            if len(sub_list) > 0:
                                wandb.log({"valid/val_sample_mse_mean": val_recovery_mean})
                        print("valid/val_loss_mean", val_loss_mean)
                        # print("valid/val_sample_mse_mean", val_recovery_mean)


                        # print()
                        # print(f"Eval and save at iter {self.step}, Val total loss: {val_loss_mean:.4f}, "
                        #       f"mse loss: {val_mse_mean:.4f}, ce loss: {val_ce_mean:.4f},"
                        #       f"  Val sample mse: {val_recovery_mean:.4f}")

                        # milestone = self.step // self.save_and_sample_every
                    if val_loss_mean < self.min_val:
                        self.min_val = val_loss_mean
                    # if val_recovery_mean < self.min_val:
                    #     self.min_val = val_recovery_mean
                        print('save model')
                        self.save()
                    torch.cuda.empty_cache()
                pbar.update(1)

        # 一次训完最后再保存，中间就不保存状态浪费时间了
        # self.save()

        # x_list = [x * self.save_and_sample_every for x in range(len(val_loss_list))]
        # # draw train-loss curves
        # plt.figure()
        # # 去除顶部和右边框框
        # ax = plt.axes()
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # plt.xlabel('epochs')  # x轴标签
        # plt.ylabel('train-loss')  # y轴标签
        # plt.plot(x_list, train_loss_list, linewidth=1, linestyle="solid", label="loss",
        #          color='blue')
        # plt.plot(x_list, train_ce_list, linewidth=1, linestyle="dotted", label=f"crossEntropy loss (*{self.config['ss_coef']})", color='green')
        # plt.plot(x_list, train_mse_list, linewidth=1, linestyle="dotted", label="MSE loss", color='red')
        # plt.legend()
        # plt.title('Train-loss Curve')
        # figure_save_path = str(self.results_folder.joinpath('figure/train-loss_curve.png'))
        # plt.savefig(figure_save_path)
        # plt.cla()
        # plt.close("all")
        #
        #
        # # draw val-loss curves
        # plt.figure()
        # # 去除顶部和右边框框
        # ax = plt.axes()
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # plt.xlabel('epochs')  # x轴标签
        # plt.ylabel('accuracy')  # y轴标签
        # plt.plot(x_list, val_accuracy_list, linewidth=1, linestyle="solid", label="average accuracy",
        #          color='blue')
        # plt.legend()
        # plt.title('val-accuracy Curve')
        # figure_save_path = str(self.results_folder.joinpath('figure/val-accuracy_curve.png'))
        # plt.savefig(figure_save_path)
        # plt.cla()
        # plt.close("all")
        #
        #
        # # draw val-loss curves
        # plt.figure()
        # # 去除顶部和右边框框
        # ax = plt.axes()
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        #
        # plt.xlabel('epochs')  # x轴标签
        # plt.ylabel('val-loss')  # y轴标签
        #
        # # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        # plt.plot(x_list, val_loss_list, linewidth=1, linestyle="solid", label="loss", color='blue')
        # plt.plot(x_list, val_ce_loss_list, linewidth=1, linestyle="dotted", label=f"crossEntropy loss (*{self.config['ss_coef']})",
        #          color='green')
        # plt.plot(x_list, val_mse_loss_list, linewidth=1, linestyle="dotted", label="MSE loss", color='red')
        # plt.legend()
        # plt.title('val-Loss curve')
        # figure_save_path = str(self.results_folder.joinpath('figure/val_loss_curve.png'))
        # plt.savefig(figure_save_path)
        # plt.show()
        print('training complete')





# used in sampling
def get_efficient_knn(model_emb, text_emb):
    # model_emb: [vocab_size, dim], text_emb: [bsz*seqlen, dim]
    emb_norm = (model_emb**2).sum(-1).view(-1, 1) # vocab, 1
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1) # d, bsz*seqlen
    arr_norm = (text_emb ** 2).sum(-1).view(-1, 1) # bsz*seqlen, 1
    # print(emb_norm.shape, arr_norm.shape)
    dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t) # (vocab, d) x (d, bsz*seqlen)
    dist = torch.clamp(dist, 0.0, np.inf)
    # print(dist.shape)
    topk_out = torch.topk(-dist, k=1, dim=0)
    return topk_out.values, topk_out.indices

# used in sampling
def denoised_fn_round(model, text_emb, t):
    # model should be the embedding layer *without grad*
    # print(text_emb.shape) # bsz, seqlen, dim+1
    model_emb = model.weight  # input_embs : [vocab_size, dim]
    # print(t)
    old_shape = text_emb.shape # bsz, seqlen, dim
    old_device = text_emb.device

    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb
    val, indices = get_efficient_knn(model_emb, text_emb.clone().to(model_emb.device))
    rounded_tokens = indices[0]
    # print(rounded_tokens.shape)
    new_embeds = model(rounded_tokens).view(*old_shape).to(old_device)
    assert new_embeds.shape == old_shape

    return new_embeds



def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



if __name__ == "__main__":
    # python3.9 train_diff.py --wandb_run_name diffusion_train
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    args = create_parser()
    print(args)
    set_seeds(args.diff_seed)
    if args.wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = args.wandb_project
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            entity=args.wandb_entity, config=vars(args)
        )

    print(f'train on {args.dataset} dataset')
    raw_dir = args.data_root
    dataset = args.dataset
    data_dir = os.path.join(raw_dir, dataset)
    ID = sorted(os.listdir(os.path.join(data_dir, 'process')))
    random.Random(4).shuffle(ID)
    val_num = int(len(ID)*args.val_ratio/(1-args.test_ratio))
    train_dataset, val_dataset = ID[:-val_num], ID[-val_num:]
    # train_ID, val_ID = ID[:256], ID[256:300]s
    # train_dataset = Cath(train_ID, data_dir)
    # val_dataset = Cath(val_ID, data_dir)
    # # test_dataset = Cath(val_ID, data_dir)

    # train_dataset_aa = Cath(train_ID, data_dir) # debug
    print(f'train on {args.dataset} dataset with {len(train_dataset)}  training data and {len(val_dataset)}  val data')

    # TODO: input dime hardcoded
    base_model = EGNN_NET2(input_dim=1280,
                          hidden_channels=args.hdim,
                          dropout=args.diff_dp,
                          n_layers=args.n_layers,
                          update_edge=True,
                          embedding_dim=args.embedding_dim,
                          norm_feat=True,
                          ss_coef=args.ss_coef,
                          ss_type_num=args.ss_type_num)

    diffusion_model = GaussianDiffusion(model=base_model,
                                        betas=get_named_beta_schedule(args.noise_schedule,
                                                                      args.diffusion_steps),
                                        ss_coef=args.ss_coef,
                                        predict_xstart=not args.pred_noise)

    trainer = Trianer(vars(args),
                     diffusion_model,
                     train_dataset=train_dataset,
                     val_dataset=val_dataset,
                     # test_dataset,
                     train_batch_size=args.diff_batch_size,
                     save_and_sample_every=1,
                     train_num_steps=args.diff_epoch*(math.ceil(len(train_dataset)/args.diff_batch_size)),   # epoch*(train_size/batch)
                     train_lr=args.diff_lr,
                     weight_decay=args.diff_wd,
                     resume_checkpoint=False)

    trainer.train()
    if args.wandb:
        wandb.finish()
