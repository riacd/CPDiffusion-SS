# site-package modification for ESM-IF
modify the sample() function 
(at~/miniconda3/envs/your_env_name/lib/python3.10/site-packages/esm/inverse_folding/gvp_transformers.py, line 88)
to the following code
```python
    def sample(self, coords, partial_seq=None, temperature=1.0, confidence=None, device=None):
        """
        Samples sequences based on multinomial sampling (no beam search).

        Args:
            coords: L x 3 x 3 list representing one backbone
            partial_seq: Optional, partial sequence with mask tokens if part of
                the sequence is known
            temperature: sampling temperature, use low temperature for higher
                sequence recovery and high temperature for higher diversity
            confidence: optional length L list of confidence scores for coordinates
        """
        L = len(coords)
        # Convert to batch format
        batch_converter = CoordBatchConverter(self.decoder.dictionary)
        batch_coords, confidence, _, _, padding_mask = (
            batch_converter([(coords, confidence, None)], device=device)
        )

        # Start with prepend token
        mask_idx = self.decoder.dictionary.get_idx('<mask>')
        sampled_tokens = torch.full((1, 1 + L), mask_idx, dtype=int)
        sampled_tokens[0, 0] = self.decoder.dictionary.get_idx('<cath>')
        if partial_seq is not None:
            for i, c in enumerate(partial_seq):
                sampled_tokens[0, i + 1] = self.decoder.dictionary.get_idx(c)

        # Save incremental states for faster sampling
        incremental_state = dict()

        # Run encoder only once
        encoder_out = self.encoder(batch_coords, padding_mask, confidence)

        # Make sure all tensors are on the same device if a GPU is present
        if device:
            sampled_tokens = sampled_tokens.to(device)

        # Decode one token at a time
        for i in range(1, L + 1):
            logits, _ = self.decoder(
                sampled_tokens[:, :i],
                encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits[0].transpose(0, 1)
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            if sampled_tokens[0, i] == mask_idx:
                sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
        sampled_seq = sampled_tokens[0, 1:]

        # Convert back to string via lookup
        return ''.join([self.decoder.dictionary.get_tok(a) for a in sampled_seq])
```