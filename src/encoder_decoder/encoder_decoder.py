import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, List, Optional
import numpy as np

def contains_nan(y: torch.Tensor):
    x = y.detach().cpu()
    for i in range(len(x.shape)):
        x=x.sum(dim=0, keepdim=False)
    return np.isnan(x)

class Encoder_Decoder(nn.Module):
    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self,
            prev_output_tokens,
            graph_batch,
            incremental_state=None,
            return_all_hiddens: bool = False,
            features_only: bool = False,
    ):
        graph_embedding = self.get_embeddings(graph_batch=graph_batch)
        encoder_out = self.prepare_enc_out(graph_embedding, graph_batch)    # {'encoder_out': [max_ss_len_in_batch, batch, 1280]}

        logits, extra = self.decoder(
            prev_output_tokens, # [batch,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
        )
        if contains_nan(graph_batch.x):
            print('graph_batch.x contains nan', graph_batch.x.shape, graph_batch.x)
        if contains_nan(graph_embedding):
            print('graph_embedding contains nan', graph_embedding.shape, graph_embedding)
        if contains_nan(logits):
            print('logits', logits.shape, logits)
        if contains_nan(prev_output_tokens):
            print('prev_output_tokens contains nan', prev_output_tokens.shape, prev_output_tokens)
        if contains_nan(encoder_out['encoder_out']):
            print('encoder_out contains nan', encoder_out['encoder_out'].shape, encoder_out['encoder_out'])

        return logits, extra

    def get_embeddings(self, graph_batch):
        graph_embedding = self.encoder(graph_batch.x, input_mask=graph_batch.input_mask)  # [batched_ss_len, 1280]
        if self.args.decoder_add_2nd_label:
            graph_embedding = torch.cat([graph_embedding, graph_batch.b_type], dim=-1)
        return graph_embedding

    def autoregressive_generation(
            self,
            graph_embedding,
            graph_batch,
            temperature=1.0,
            max_len=1024
    ):
        encoder_out = self.prepare_enc_out(graph_embedding, graph_batch)

        # Save incremental states for faster sampling
        incremental_state = dict()

        # Start with prepend token
        batch_size = graph_batch['ptr'].size(0) - 1
        bos_id = self.decoder.tokenizer.token_to_id('<cls>')
        sampled_tokens = torch.full((batch_size, 1), bos_id, dtype=int, device=graph_embedding.device)
        invalid_tokens = ['.', '-', '<null_1>', '<mask>', '<unk>', '<pad>']
        invalid_ids = self.decoder.tokenizer.convert_tokens_to_ids(invalid_tokens)

        # Decode one token at a time
        for i in range(1, 1 + max_len):
            logits, _ = self.decoder(
                sampled_tokens[:, :i],
                encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits[:, -1, :] / temperature  # Take the last step
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # Multinomial sampling    # (batch_size, 1)
            valid_flag = True
            for token_id in next_token.view(-1):
                if token_id in invalid_ids:
                    valid_flag = False
                    break
            if valid_flag:
                sampled_tokens = torch.cat([sampled_tokens, next_token], dim=-1)    # (batch_size, n) -> (batch_size, n+1)

        return sampled_tokens

    def prepare_enc_out(self, graph_embedding, graph_batch):
        """
        reshape graph_embedding to diff_out_batch

        Args:
            graph_embedding: [node_num, feature_size]
            graph_batch: x, edge_index, edge_attr, ptr

        Returns:
            encoder_out: [max_node_num, batch_size, feature_size]
            encoder_padding_mask: [batch_size, max_node_num]
        """
        node_num_per_graph = graph_batch['ptr'][1:] - graph_batch['ptr'][:-1]
        max_node_num = torch.max(node_num_per_graph)
        batch_size = graph_batch['ptr'].size(0) - 1
        feature_size = graph_batch['x'].size(-1)
        diff_out_batch = torch.zeros(batch_size, max_node_num, feature_size, device=graph_embedding.device)
        encoder_padding_mask = torch.ones(batch_size, max_node_num, device=graph_embedding.device)

        for i in range(batch_size):
            start_ptr = graph_batch['ptr'][i].item()
            end_ptr = graph_batch['ptr'][i + 1].item()
            diff_out_batch[i, :end_ptr - start_ptr] = graph_embedding[start_ptr:end_ptr]
            encoder_padding_mask[i, :end_ptr - start_ptr] = 0
        encoder_out = {
            'encoder_out': diff_out_batch.transpose(0, 1),
            'encoder_padding_mask': encoder_padding_mask
        }
        return encoder_out
