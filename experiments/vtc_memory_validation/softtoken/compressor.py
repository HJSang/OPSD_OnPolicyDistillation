"""
Minimal soft-token compressor prototype (gated-pooling, ICAE/Glyph-style).

Goal of this prototype: prove the pipeline "compress N token embeddings -> k
continuous vectors -> inject into a FROZEN decoder -> reconstruct / answer".
Not optimized; single-block (no per-session chunking yet).

Design (making 2: gated average pooling):
  - Encoder = the first `enc_layers` transformer layers of the frozen decoder
    (shares the decoder's embedding space -> no alignment problem, cheap).
  - A small linear adapter + a learned gate mix the encoder hidden states with
    the raw token embeddings, then average-pool every `factor` tokens into one
    soft token:
        alpha = sigmoid(gate(h))                      # (N,1)
        fused = alpha * adapter(h) + (1-alpha) * emb   # (N, d)
        soft  = avg_pool(fused, factor)                # (N/factor, d)
  - Only {adapter, gate} (and optionally the borrowed encoder layers) are
    trained. The decoder is frozen.

The soft tokens are fed to the frozen decoder via `inputs_embeds`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTokenCompressor(nn.Module):
    def __init__(self, decoder, factor=8, enc_layers=2, train_encoder=False):
        """
        decoder: a loaded HF CausalLM (frozen). Must expose
                 .model.embed_tokens and .model.layers (Qwen2/Llama-style).
        factor:  compression ratio (N -> N/factor soft tokens).
        enc_layers: how many of the decoder's bottom layers to reuse as encoder.
        """
        super().__init__()
    def __init__(self, decoder, factor=8, enc_layers=2, train_encoder=False,
                 mode="simple", role_factors=None):
        """
        decoder: a loaded HF CausalLM (frozen). Must expose
                 .model.embed_tokens and .model.layers (Qwen2/Llama-style).
        factor:  compression ratio (N -> N/factor soft tokens).
        enc_layers: how many of the decoder's bottom layers to reuse as encoder.
        mode:    "simple" -> one uniform gated pooling over the whole input
                            (the current prototype).
                 "full"   -> per-segment pooling (each session/turn pooled
                            independently) with per-role compression factors,
                            i.e. our method: compress user facts lightly,
                            assistant chatter heavily. Requires `segments`.
        role_factors: dict like {"user": 4, "assistant": 16} used in "full"
                 mode to set each segment's compression factor by role.
        """
        super().__init__()
        self.decoder = decoder
        self.factor = factor
        self.enc_layers = enc_layers
        self.mode = mode
        self.role_factors = role_factors or {"user": 4, "assistant": 16}
        d = decoder.config.hidden_size

        # Encoder = the decoder's bottom layers. If we're going to TRAIN them we
        # must DEEP-COPY, otherwise we'd be mutating the frozen decoder's own
        # layers (they share module references) and corrupt reconstruction.
        base = decoder.model
        self.embed_tokens = base.embed_tokens          # frozen (shared, not trained)
        self.rotary_emb = getattr(base, "rotary_emb", None)
        if train_encoder:
            import copy
            self.layers = copy.deepcopy(
                nn.ModuleList(base.layers[:enc_layers]))
        else:
            self.layers = base.layers[:enc_layers]     # borrowed (frozen)

        # trainable head
        self.adapter = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, 1)

        # freeze decoder
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        if train_encoder:
            for p in self.layers.parameters():
                p.requires_grad_(True)

        # init adapter near identity, gate near 0 (start ~ raw embeddings)
        nn.init.eye_(self.adapter.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)  # sigmoid(-2)~0.12

        # keep the trainable head in fp32 for stable optimization; cast its
        # inputs/outputs around the frozen bf16 decoder as needed.
        self.adapter.to(torch.float32)
        self.gate.to(torch.float32)

    @property
    def device(self):
        return next(self.adapter.parameters()).device

    def load_trained(self, ckpt_path, map_location="cuda"):
        """Load a checkpoint saved by train.py (adapter/gate [+ enc_layers])."""
        ck = torch.load(ckpt_path, map_location=map_location)
        self.adapter.load_state_dict(ck["adapter"])
        self.gate.load_state_dict(ck["gate"])
        if "enc_layers" in ck:
            self.layers.load_state_dict(ck["enc_layers"])
        return ck.get("args", {})

    def _fuse(self, input_ids, attention_mask):
        """Embed -> borrowed encoder layers -> gated fuse. Returns fused (B,N,d)."""
        emb = self.embed_tokens(input_ids)             # (B, N, d)
        hidden = emb
        pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device
                               ).unsqueeze(0).expand(input_ids.shape[0], -1)
        pos_emb = self.rotary_emb(hidden, pos_ids) if self.rotary_emb else None
        causal = _make_causal_mask(attention_mask, hidden.dtype)
        for layer in self.layers:
            out = layer(hidden, attention_mask=causal,
                        position_ids=pos_ids, position_embeddings=pos_emb)
            hidden = out[0] if isinstance(out, tuple) else out

        alpha = torch.sigmoid(self.gate(hidden.float()))        # (B, N, 1) fp32
        fused = alpha * self.adapter(hidden.float()) + (1 - alpha) * emb.float()
        return fused.to(emb.dtype)

    def encode(self, input_ids, attention_mask=None, segments=None):
        """input_ids: (B, N) -> soft tokens.

        simple mode: uniform pooling every `factor` tokens -> (B, ceil(N/factor), d).
        full mode:   per-segment pooling. `segments` is a list (len B) of lists
                     of (start, end, role) spans; each span is pooled with
                     role_factors[role]. Returns a list (len B) of (M_i, d)
                     tensors (variable length per item).
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        fused = self._fuse(input_ids, attention_mask)  # (B, N, d)

        if self.mode == "simple" or segments is None:
            return _avg_pool_tokens(fused, self.factor)  # (B, M, d)

        # full mode: pool each session/turn segment independently by role.
        # If a role's factor == 1 we keep the RAW token embeddings (bypass the
        # encoder/adapter/gate) so those spans are truly lossless -> use this to
        # preserve user facts verbatim while compressing assistant chatter.
        raw_emb = self.embed_tokens(input_ids)         # (B, N, d)
        out = []
        for b, segs in enumerate(segments):
            parts = []
            for (start, end, role) in segs:
                f = self.role_factors.get(role, self.factor)
                if f == 1:
                    parts.append(raw_emb[b, start:end])          # (L, d) lossless
                else:
                    seg = fused[b, start:end].unsqueeze(0)       # (1, L, d)
                    parts.append(_avg_pool_tokens(seg, f)[0])    # (ceil(L/f), d)
            out.append(torch.cat(parts, dim=0) if parts
                       else fused[b, :0])
        return out  # list of (M_i, d)

    def forward_with_soft_list(self, soft_list, target_ids, target_mask=None):
        """Full-mode decode for B=1: soft_list is [ (M,d) ]. Feeds
        [soft ; target_emb] and returns logits over target positions."""
        soft = soft_list[0].unsqueeze(0)               # (1, M, d)
        return self.forward_with_soft(soft, target_ids, target_mask)

    def forward_with_soft(self, soft, target_ids, target_mask=None):
        """Teacher-forced decode: feed [soft ; target_emb] and get logits over
        the target positions. Returns logits (B, T, V)."""
        tgt_emb = self.embed_tokens(target_ids)        # (B, T, d)
        inp = torch.cat([soft, tgt_emb], dim=1)        # (B, M+T, d)
        M = soft.shape[1]
        attn = torch.ones(inp.shape[:2], device=inp.device, dtype=torch.long)
        if target_mask is not None:
            attn[:, M:] = target_mask
        out = self.decoder(inputs_embeds=inp, attention_mask=attn)
        logits = out.logits[:, M:, :]                  # only target positions
        return logits


def _avg_pool_tokens(x, factor):
    """(B, N, d) -> (B, ceil(N/factor), d) via non-overlapping mean pooling."""
    B, N, d = x.shape
    pad = (factor - N % factor) % factor
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    x = x.view(B, -1, factor, d).mean(dim=2)
    return x


def _make_causal_mask(attention_mask, dtype):
    """(B, N) 1/0 mask -> (B,1,N,N) additive causal+padding mask."""
    B, N = attention_mask.shape
    min_val = torch.finfo(dtype).min
    causal = torch.full((N, N), min_val, dtype=dtype,
                        device=attention_mask.device).triu(1)
    causal = causal[None, None, :, :].expand(B, 1, N, N).clone()
    pad = (1 - attention_mask[:, None, None, :].to(dtype)) * min_val
    return causal + pad


def build_role_segments(tokenizer, turns, max_len=None):
    """Tokenize a conversation's turns and return (input_ids, segments).

    turns: list of {"role": "user"/"assistant", "content": str}.
    Returns:
        input_ids: (1, N) long tensor
        segments:  list with ONE element (for B=1): list of (start, end, role)
                   spans marking each turn's token range -> used by full mode
                   to pool each turn independently with its role's factor.
    """
    ids, spans = [], []
    for turn in turns:
        role = "user" if turn.get("role") in ("user", "human") else "assistant"
        text = f"{role}: {turn.get('content', '')}\n"
        tok = tokenizer(text, add_special_tokens=False)["input_ids"]
        start = len(ids)
        ids.extend(tok)
        spans.append((start, len(ids), role))
        if max_len and len(ids) >= max_len:
            ids = ids[:max_len]
            spans[-1] = (spans[-1][0], max_len, spans[-1][2])
            break
    input_ids = torch.tensor([ids], dtype=torch.long)
    return input_ids, [spans]

