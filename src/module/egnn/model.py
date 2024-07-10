import torch
import math
import torch_geometric
import torch.nn.functional as F
from torch import nn

from src.data.cath_2nd import aa_vocab
from src.module.egnn.egnn import EGNN_Sparse

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.view(emb.shape[0],-1)





# class EGNN_NET(nn.Module):
#     """
#         Use noise x in egnn propagation.
#     """
#     def __init__(self, max_b_aa_num, hidden_channels, dropout, n_layers, output_dim=21,
#                   embedding_dim=64, update_edge=True, norm_feat=False):
#         super(EGNN_NET, self).__init__()
#         torch.manual_seed(12345)
#         self.dropout = dropout
#
#         self.update_edge = update_edge
#         self.mpnn_layes = nn.ModuleList()
#         self.time_mlp_list = nn.ModuleList()
#         self.ff_list = nn.ModuleList()
#         self.ff_norm_list = nn.ModuleList()
#         self.sinu_pos_emb_time = SinusoidalPosEmb(hidden_channels)
#         self.sinu_pos_emb_aapos = SinusoidalPosEmb(hidden_channels)
#         self.n_layers = n_layers
#         self.output_dim = output_dim
#         self.max_b_aa_num = max_b_aa_num
#
#         self.time_mlp = nn.Sequential(self.sinu_pos_emb_time, nn.Linear(hidden_channels, hidden_channels), nn.SiLU(),
#                                       nn.Linear(hidden_channels, embedding_dim))
#
#         for i in range(n_layers):
#             if i == 0:
#                 layer = EGNN_Sparse(embedding_dim*2, m_dim=hidden_channels, hidden_dim=hidden_channels,
#                                     out_dim=hidden_channels, edge_attr_dim=embedding_dim, dropout=dropout,
#                                     update_edge=self.update_edge, norm_feats=norm_feat)
#             else:
#                 layer = EGNN_Sparse(hidden_channels, m_dim=hidden_channels, hidden_dim=hidden_channels,
#                                     out_dim=hidden_channels, edge_attr_dim=embedding_dim, dropout=dropout,
#                                     update_edge=self.update_edge, norm_feats=norm_feat)
#
#             time_mlp_layer = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, (hidden_channels) * 2))
#             ff_norm = torch_geometric.nn.norm.LayerNorm(hidden_channels)
#             ff_layer = nn.Sequential(nn.Linear(hidden_channels, hidden_channels * 4), nn.Dropout(p=dropout), nn.GELU(),
#                                      nn.Linear(hidden_channels * 4, hidden_channels))
#
#             self.mpnn_layes.append(layer)
#             self.time_mlp_list.append(time_mlp_layer)
#             self.ff_list.append(ff_layer)
#             self.ff_norm_list.append(ff_norm)
#
#         # TODO: here hard-code dimension of the input to be self.ss_type_num for b_type (sstype) and 1 for edge (dist)
#         self.ss_embedding = nn.Sequential(nn.Linear(self.ss_type_num, hidden_channels), nn.SiLU(),
#                                     nn.Linear(hidden_channels, embedding_dim))
#         self.edge_embedding = nn.Linear(1, embedding_dim)
#
#         #### token embed and decode
#         # TODO: hard-code vocab size to be 21 (with pad token)
#         # TODO: chane to esm2 embedding
#         # TODO: add positional encoding?
#         self.aa_embedding = nn.Embedding(21, embedding_dim, padding_idx = aa_vocab['PAD'])
#         # self.aa_pos_embed = nn.Sequential(self.sinu_pos_emb_aapos, nn.Linear(hidden_channels, hidden_channels), nn.SiLU(),
#         #                               nn.Linear(hidden_channels, embedding_dim))
#         self.aa_mlp = nn.Sequential(nn.Linear(embedding_dim, hidden_channels), nn.SiLU(),
#                                      nn.Linear(hidden_channels, embedding_dim)) # this should be called after reshaping
#         self.aa_decode = nn.Sequential(nn.Linear(hidden_channels*self.max_b_aa_num, hidden_channels), nn.SiLU(),
#                                        nn.Linear(hidden_channels, embedding_dim*self.max_b_aa_num)
#                                        )# decode the aa seq
#         # do not train get_logits
#         self.get_logits = nn.Linear(embedding_dim, 21)
#         # with torch.no_grad():
#         #     self.get_logits.weight = self.aa_embedding.weight
#         self.get_logits.weight.data = self.aa_embedding.weight.data.clone()
#
#     def forward(self, data, time):
#         # x is [ss_len, aa_len, embedding_dim] cat, b_pos is [ss_len, 3], b_edge_index [2, ss_edge], b_edge_attr [ss_edge,1], b_type [ss_len, self.ss_type_num]
#         # time [B,]
#         # note the size of ss_len aggregates the ss_len from all batch
#         x, b_pos, b_edge_index, b_edge_attr, b_type, batch = data.x, data.b_pos, data.b_edge_index, data.b_edge_attr, data.b_type, data.batch
#
#         # x_pos_embed = self.aa_pos_embed(torch.arange(self.max_b_aa_num).to(x.device)).unsqueeze(0).expand(x.shape) # [ss_len, aa_len, embedding_dim]
#         # x_embed = self.aa_mlp(torch.cat((x,x_pos_embed), dim=-1).view(x.shape[0], -1))  # [ss_len, embedding_dim]
#         # x_embed = self.aa_mlp(x.view(x.shape[0], -1))
#         x_embed = self.aa_mlp(x).mean(dim=-2)
#
#         t = self.time_mlp(time)  # [B, embedding_dim]
#         b_embed = self.ss_embedding(b_type.float())  # [ss_len, embedding_dim]
#         b_edge_attr = self.edge_embedding(b_edge_attr)
#         x = torch.cat([b_pos, b_embed, x_embed], dim=1)
#
#         for i, layer in enumerate(self.mpnn_layes):
#             # GNN aggregate
#             if self.update_edge:
#                 h, b_edge_attr = layer(x, b_edge_index, b_edge_attr, batch)  # [ss_len,hidden_dim]
#             else:
#                 h = layer(x, b_edge_index, b_edge_attr, batch)  # [ss_len,hidden_dim]
#
#             # time and conditional shift
#             corr, feats = h[:, 0:3], h[:, 3:]
#             time_emb = self.time_mlp_list[i](t)  # [B,hidden_dim*2]
#             scale_, shift_ = time_emb.chunk(2, dim=1)
#             scale = scale_[data.batch]
#             shift = shift_[data.batch]
#             feats = feats * (scale + 1) + shift
#
#             # FF neural network
#             feature_norm = self.ff_norm_list[i](feats, batch)
#             feats = self.ff_list[i](feature_norm) + feature_norm
#
#             x = torch.cat([corr, feats], dim=-1)
#
#         corr, x = x[:, 0:3], x[:, 3:]
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         # upsampling
#         # TODO: double check here
#         x = x.unsqueeze(-2).expand(x.shape[0], self.max_b_aa_num, x.shape[1]).reshape(x.shape[0], -1) # [ss_len, hidden_dim*aa_len]
#         x = self.aa_decode(x).view(*data.x.shape)
#         # output is [ss_len, aa_len, embedding_dim]
#         return x





class EGNN_NET2(nn.Module):
    """
    """
    def __init__(self, input_dim, hidden_channels, dropout, n_layers, output_dim=None,
                  embedding_dim=64, update_edge=True, norm_feat=False, ss_coef=0.1, ss_type_num=3):
        super(EGNN_NET2, self).__init__()
        """
            input_dim : the input encoding dim for ss (depending on the encoder)
        """
        torch.manual_seed(12345)
        self.dropout = dropout

        self.update_edge = update_edge
        self.mpnn_layes = nn.ModuleList()
        self.time_mlp_list = nn.ModuleList()
        self.ff_list = nn.ModuleList()
        self.ff_norm_list = nn.ModuleList()
        self.sinu_pos_emb_time = SinusoidalPosEmb(hidden_channels)

        self.n_layers = n_layers
        self.input_dim = input_dim
        self.ss_coef = ss_coef
        self.ss_type_num = ss_type_num
        if output_dim is None:
            self.output_dim = input_dim

        self.time_mlp = nn.Sequential(self.sinu_pos_emb_time, nn.Linear(hidden_channels, hidden_channels), nn.SiLU(),
                                      nn.Linear(hidden_channels, embedding_dim))

        for i in range(n_layers):
            if i == 0:
                layer = EGNN_Sparse(2*embedding_dim, m_dim=hidden_channels, hidden_dim=hidden_channels,
                                    out_dim=hidden_channels, edge_attr_dim=embedding_dim, dropout=dropout,
                                    update_edge=self.update_edge, norm_feats=norm_feat)
            else:
                layer = EGNN_Sparse(hidden_channels, m_dim=hidden_channels, hidden_dim=hidden_channels,
                                    out_dim=hidden_channels, edge_attr_dim=embedding_dim, dropout=dropout,
                                    update_edge=self.update_edge, norm_feats=norm_feat)

            time_mlp_layer = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, (hidden_channels) * 2))
            ff_norm = torch_geometric.nn.norm.LayerNorm(hidden_channels)
            ff_layer = nn.Sequential(nn.Linear(hidden_channels, hidden_channels * 4), nn.Dropout(p=dropout), nn.GELU(),
                                     nn.Linear(hidden_channels * 4, hidden_channels))

            self.mpnn_layes.append(layer)
            self.time_mlp_list.append(time_mlp_layer)
            self.ff_list.append(ff_layer)
            self.ff_norm_list.append(ff_norm)

        # TODO: here hard-code dimension of the input to be self.ss_type_num for b_type (sstype) and 1 for edge (dist)
        self.ss_embedding = nn.Sequential(nn.Linear(self.ss_type_num, hidden_channels), nn.SiLU(),
                                    nn.Linear(hidden_channels, embedding_dim))
        self.edge_embedding = nn.Linear(1, embedding_dim)

        #### token embed and decode
        # TODO: hard-code vocab size to be 21 (with pad token)
        # TODO: chane to esm2 embedding
        # TODO: add positional encoding?
        # self.aa_embedding = nn.Embedding(21, embedding_dim, padding_idx = aa_vocab['PAD'])
        # self.aa_pos_embed = nn.Sequential(self.sinu_pos_emb_aapos, nn.Linear(hidden_channels, hidden_channels), nn.SiLU(),
        #                               nn.Linear(hidden_channels, embedding_dim))
        self.ss_mlp = nn.Sequential(nn.Linear(input_dim, hidden_channels), nn.SiLU(),
                                     nn.Linear(hidden_channels, embedding_dim))
        self.ss_decode = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.SiLU(),
                                       nn.Linear(hidden_channels, (input_dim + self.ss_type_num) if ss_coef > 0 else input_dim)
                                       )# decode the aa seq
        # do not train get_logits
        # self.get_logits = nn.Linear(embedding_dim, 21)
        # with torch.no_grad():
        #     self.get_logits.weight = self.aa_embedding.weight
        # self.get_logits.weight.data = self.aa_embedding.weight.data.clone()

        # ss position embed
        # self.sinu_pos_emb_sspos = SinusoidalPosEmb(hidden_channels)
        # # self.ss_pos_embed = nn.Sequential(self.sinu_pos_emb_sspos, nn.Linear(hidden_channels, embedding_dim), nn.SiLU(),
        # #                                nn.Linear(embedding_dim, hidden_channels))
        # self.ss_pos_embed = nn.Sequential(self.sinu_pos_emb_sspos, nn.Linear(hidden_channels, hidden_channels))

    def forward(self, data, time):
        # x is [ss_len, input_dim] cat, b_pos is [ss_len, 3], b_edge_index [2, ss_edge], b_edge_attr [ss_edge,1], b_type [ss_len, self.ss_type_num]
        # time [B,]
        # note the size of ss_len aggregates the ss_len from all batch
        x, b_pos, b_edge_index, b_edge_attr, b_type, batch = data.x, data.b_pos, data.b_edge_index, data.b_edge_attr, data.b_type, data.batch
        ptr = data.ptr

        # position embedding of ss
        # positions = torch.cat([torch.arange(0, idx.item()) for idx in torch.diff(ptr)], dim=0)
        # x_pos_embed = self.ss_pos_embed(positions.to(x.device)) # [ss_len, embedding_dim]
        # x_embed = self.aa_mlp(torch.cat((x,x_pos_embed), dim=-1).view(x.shape[0], -1))  # [ss_len, embedding_dim]
        # x_embed = self.aa_mlp(x.view(x.shape[0], -1))

        x_embed = self.ss_mlp(x) # [ss_len, embedding_dim]

        t = self.time_mlp(time)  # [B, embedding_dim]
        b_embed = self.ss_embedding(b_type.float())  # [ss_len, embedding_dim]
        b_edge_attr = self.edge_embedding(b_edge_attr)
        x = torch.cat([b_pos, b_embed, x_embed], dim=1)

        for i, layer in enumerate(self.mpnn_layes):
            # GNN aggregate
            if self.update_edge:
                h, b_edge_attr = layer(x, b_edge_index, b_edge_attr, batch)  # [ss_len,hidden_dim]
            else:
                h = layer(x, b_edge_index, b_edge_attr, batch)  # [ss_len,hidden_dim]

            # time and conditional shift
            corr, feats = h[:, 0:3], h[:, 3:]
            time_emb = self.time_mlp_list[i](t)  # [B,hidden_dim*2]
            scale_, shift_ = time_emb.chunk(2, dim=1)
            scale = scale_[data.batch]
            shift = shift_[data.batch]
            feats = feats * (scale + 1) + shift

            # FF neural network
            feature_norm = self.ff_norm_list[i](feats, batch)
            feats = self.ff_list[i](feature_norm) + feature_norm

            x = torch.cat([corr, feats], dim=-1)

        corr, x = x[:, 0:3], x[:, 3:]
        x = F.dropout(x, p=self.dropout, training=self.training)
        # upsampling
        # x = x.unsqueeze(-2).expand(x.shape[0], self.max_b_aa_num, x.shape[1]).reshape(x.shape[0], -1) # [ss_len, hidden_dim*aa_len]
        x = self.ss_decode(x)
        # assert data.x.shape == x.shape
        # output is [ss_len, input_dim]

        if self.ss_coef > 0:
            # add softmax to ss_type prediction output
            return x[...,:self.input_dim], x[...,self.input_dim:]
        else:
            assert data.x.shape == x.shape
            return x, None