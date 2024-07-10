import argparse
import os

# def get_args():
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--Date', type=str, default='Debug',
#                         help='Date of experiment')
#
#     parser.add_argument('--dataset', type=str, default='CATH',
#                         help='which dataset used for training, CATH or TS')
#     parser.add_argument('--data_dir', type=str, default='data',
#                         help='which dataset used for training, CATH or TS')
#
#     parser.add_argument('--test_num', type=int, default=200,
#                         help='Number of test samples')
#     parser.add_argument('--batch_size', type=int, default=100,
#                         help='Training batch size')
#
#     parser.add_argument('--embedding_dim', type=int, default=128,
#                         help='embedding dim')
#     parser.add_argument('--n_layers', type=int, default=2,
#                         help='number of egnn layers')
#     parser.add_argument('--max_b_aa_num', type=int, default=100,
#                         help='maximum length for aa seq per ss block')
#     parser.add_argument('--hdim', type=int, default=128,
#                         help='hidden dimension')
#     parser.add_argument('--ss_coef', type=float, default=0.025,
#                         help='coefficient of ce loss for predicting ss')
#
#     parser.add_argument('--noise_schedule', type=str, default='cosine',
#                         help='Noise schedule for betas')
#     parser.add_argument('--diffusion_steps', type=int, default=500,
#                         help='number of diffusion steps')
#
#     parser.add_argument('--epoch', type=int, default=300,
#                         help='Epoch')
#     parser.add_argument('--lr', type=float, default=1e-4,
#                         help='Learning rate')
#     parser.add_argument('--wd', type=float, default=1e-5,
#                         help='weight decay')
#     parser.add_argument('--dp', type=float, default=0.2,
#                         help='Dropout ratio')
#
#     parser.add_argument('--seed', type=int, default=1001,
#                         help='Random seed')
#
#     args = parser.parse_args()
#     config = vars(args)
#     return config


def create_parser():
    parser = argparse.ArgumentParser()


    # decoder hyperparameters
    parser.add_argument("--encoder_embed_dim", type=int, default=1280, help="encoder embedding dimension")
    parser.add_argument("--decoder_layers", type=int, default=3, help="number of decoder layers")
    parser.add_argument("--decoder_embed_dim", type=int, default=1280, help="decoder hidden dimension")
    parser.add_argument("--decoder_ffn_embed_dim", type=int, default=4960, help="decoder feedforward hidden dimension")
    parser.add_argument("--decoder_attention_heads", type=int, default=8, help="decoder number of heads")
    parser.add_argument("--attention_dropout", type=float, default=0.1,
                        help="dropout probability for attention weights")
    parser.add_argument("--decoder_dropout", type=float, default=0.0, help="dropout probability for decoder")

    # diffusion hyperparameters
    parser.add_argument('--embedding_dim', type=int, default=1280, help='embedding dim')
    parser.add_argument('--n_layers', type=int, default=6, help='number of egnn layers')
    parser.add_argument('--max_b_aa_num', type=int, default=100, help='maximum length for aa seq per ss block')
    parser.add_argument('--hdim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--ss_coef', type=float, default=0, help='coefficient of ce loss for predicting ss')
    parser.add_argument('--noise_schedule', type=str, default='cosine', help='Noise schedule for betas')
    parser.add_argument('--diffusion_steps', type=int, default=500, help='number of diffusion steps')
    parser.add_argument('--pred_noise', action='store_true', help='whether to predict noise', default=False)

    # train decoder & encoder model
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--ckpt_root", type=str, default="results/decoder/ckpt", help="dir to store weights of encoder & decoder")
    parser.add_argument('--ckpt_dir', default=None, help='directory to save trained models')
    parser.add_argument('--model_name', default="model.pt", help='model name')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument('--max_train_epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=20, help='Training batch size')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    # parser.add_argument('--train_with_raw_ESM2_embedding', action='store_true', help='train without encoder output', default=False)

    # train diffusion model
    parser.add_argument('--diff_batch_size', type=int, default=50, help='Training/Sampling batch size')
    parser.add_argument('--diff_epoch', type=int, default=20, help='Epoch')
    parser.add_argument('--diff_lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--diff_wd', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--diff_dp', type=float, default=0.2, help='Dropout ratio')
    parser.add_argument('--diff_seed', type=int, default=1001, help='Random seed')
    parser.add_argument('--diff_model_name', type=str, default='diffusion_model', help='name of trained diffusion model (no appendix)')


    # encoder (embedding pooling) settings
    # encoder weights are stored with decoder
    parser.add_argument("--encoder_type", type=str, default='AttentionPooling', help="type of embedding pooling")

    # sample sequences
    parser.add_argument('--sample', action='store_true', help='sample / tra'
                                                              'in', default=False)
    parser.add_argument('--sample_out_split', action='store_true', help='sample output in multiple/one .fasta file', default=False) # must output multiple files if using colabfold
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for sampling')
    parser.add_argument('--sample_nums', type=int, default=100, help='numbers of generated samples for each ss input')
    parser.add_argument('--max_len', type=int, default=1024, help='max generation length')
    parser.add_argument("--decoder_ckpt", type=str, default=None)
    parser.add_argument("--diff_ckpt", type=str, default=None)  # also used in decoder training
    parser.add_argument('--sample_out_dir', default="results/sample", help='directory to save sampled seqs (this dir will be overwirtten)')
    # filter for sampling
    parser.add_argument('--min_len_filter', type=float, default=0, help='remove generated seqs with gen_len/ori_len < min_len')
    parser.add_argument('--max_len_filter', type=float, default=100, help='remove generated seqs with gen_len/ori_len > max_len')
    # diffusion sampling
    parser.add_argument('--clip_denoised', action='store_true', help='clip sampled outputs between -1 and 1', default=False)
    parser.add_argument('--clamp_step', type=int, default=10, help='Steps start or end to apply denoised_fn')
    parser.add_argument('--clamp_first', action='store_true', default=False, help='clamp start or end')
    parser.add_argument('--apply_denoised_fn', action='store_true', default=False, help='whether to apply denoised_fn')
    # sampling mode
    parser.add_argument('--random_embedding', type=bool, default=False, help='use random embedding as decoder input for seq generation')





    # dataset
    parser.add_argument('--data_root', type=str, default='data', help='directory storing all the datasets')
    parser.add_argument('--dataset', type=str, default='CATH', help='dataset name (CATH / AFDB / sample)')
    # parser.add_argument('--val_num', type=int, default=6000, help='Number of test samples')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation samples')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test samples')
    parser.add_argument('--max_data_num', type=int, default=1000000000, help='Max number of data(train+val for training process/ test for sampling process) selected')
    parser.add_argument('--diff_dir', type=str, default=None, help='dir for graph data')
    parser.add_argument('--deco_dir', type=str, default=None, help='dir for graph data')



    # dataset process
    parser.add_argument('--ESM2_dir', type=str, default='models/esm2_t33_650M_UR50D', help='model name')
    parser.add_argument('--max_process_pdb_num', type=int, default=1000000000)
    parser.add_argument('--split_with_ss8', action='store_true', help='split embedding using 8 types of second structures, '
                                                                      'using 3 types of ss by default', default=False)
    parser.add_argument('--ss_type_num', type=int, help='split embedding using 8 types of second structures, '
                                                                      'using 3 types of ss by default', default=3)




    # wandb log
    parser.add_argument('--wandb', action='store_true', help='use wandb to log training', default=False)
    parser.add_argument('--wandb_project', type=str, default="Prodiff_2nd")
    parser.add_argument("--wandb_entity", type=str, default="matwings")
    parser.add_argument('--wandb_run_name', type=str, default=None)

    # special configuration
    parser.add_argument('--decoder_add_2nd_label', action='store_true',
                        help='add 2nd structure labels to input embeddings for decoder', default=False)
    parser.add_argument('--max_len_per_ss', type=int, default=1000, help='aa length in one second structure threshold')

    # benchmark results
    parser.add_argument('--seq_dir', type=str, default='benchmark/seqs', help='directory storing generated seqs')
    parser.add_argument('--structure_dir', type=str, default='benchmark/structures', help='directory storing generated structures')
    parser.add_argument('--random_seed', type=int, default=2523455, help='set random seed to synchronize random shuffle in benchmark')


    # esmfold
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--fasta_file", type=str, default=None)
    parser.add_argument("--fasta_chunk_num", type=int, default=None)
    parser.add_argument("--fasta_chunk_id", type=int, default=None)
    parser.add_argument("--fasta_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--out_file", type=str, default="result.pdb")
    parser.add_argument("--out_info_file", type=str, default=None)
    parser.add_argument("--fold_chunk_size", type=int)

    # metrics
    parser.add_argument("--metrics_model", type=str, default='ProtDiffS40')




    args = parser.parse_args()
    if args.decoder_add_2nd_label:
        args.encoder_embed_dim = 1288

    args.diff_dir = os.path.join(args.data_root, args.dataset, 'process')
    args.deco_dir = os.path.join(args.data_root, args.dataset, 'aa_feature')
    if args.split_with_ss8:
        args.ss_type_num = 8
    else:
        args.ss_type_num = 3
    if args.diff_model_name == 'diffusion_model':
        if args.wandb_run_name:
            args.diff_model_name = args.wandb_run_name
    if args.model_name == 'model.pt':
        if args.wandb_run_name:
            args.model_name = args.wandb_run_name+'.pt'


    # path


    # args.model_name = f"_dataset={self.config['dataset']}_result_lr={self.config['lr']}_wd={self.config['wd']}_dp={self.config['dp']}_hidden={self.config['hdim']}_noisy_type={self.config['noise_schedule']}"
    return args
