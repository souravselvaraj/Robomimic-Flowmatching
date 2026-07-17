"""
Transformer velocity network for Conditional Flow Matching.

`ConditionalTransformer1D` is a 1D DiT (diffusion-transformer) over the predicted
action chunk: the `Tp` action steps are treated as tokens, and the flow time `t`
plus the encoded observation `global_cond` are injected through adaptive
LayerNorm (AdaLN-Zero, Peebles & Xie 2023). It is a drop-in alternative to
`robomimic.models.diffusion_policy_nets.ConditionalUnet1D`, exposing the exact
same call signature so the rest of the flow-matching algorithm is unchanged:

    velocity = net(sample=[B, Tp, Da], timestep=[B], global_cond=[B, C])  -> [B, Tp, Da]
"""
import torch
import torch.nn as nn

from robomimic.models.diffusion_policy_nets import SinusoidalPosEmb


def modulate(x, shift, scale):
    """AdaLN modulation. x: [B, T, D]; shift, scale: [B, D]."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """A DiT block: AdaLN-modulated self-attention + MLP, with gated residuals."""

    def __init__(self, n_emb, n_head, p_drop, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(n_emb, n_head, dropout=p_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)
        hidden = int(n_emb * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, hidden), nn.GELU(), nn.Dropout(p_drop), nn.Linear(hidden, n_emb)
        )
        # produces (shift1, scale1, gate1, shift2, scale2, gate2) from the conditioning vector
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(n_emb, 6 * n_emb))
        # AdaLN-Zero: start each block as an identity function
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(self, x, c, attn_mask=None):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada_ln(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn_out
        h = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        return x


class ConditionalTransformer1D(nn.Module):
    """
    1D DiT velocity field. Drop-in replacement for ConditionalUnet1D.

    Args:
        input_dim (int): action dimension Da.
        global_cond_dim (int): dimension of the flattened observation conditioning
            (obs_horizon * obs_dim); may be 0/None for an unconditional model.
        n_emb (int): transformer embedding width.
        n_layer (int): number of DiT blocks.
        n_head (int): attention heads.
        p_drop (float): dropout probability.
        max_positions (int): maximum action-chunk length (prediction horizon).
        diffusion_step_embed_dim (int): width of the sinusoidal time embedding.
        causal (bool): if True, mask attention to be left-to-right over the chunk.
    """

    def __init__(
        self,
        input_dim,
        global_cond_dim,
        n_emb=256,
        n_layer=8,
        n_head=4,
        p_drop=0.1,
        max_positions=16,
        diffusion_step_embed_dim=256,
        causal=False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_positions, n_emb))
        self.drop = nn.Dropout(p_drop)

        # flow time -> conditioning vector
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb),
        )
        # observation conditioning -> added to the time vector
        self.has_cond = global_cond_dim is not None and global_cond_dim > 0
        if self.has_cond:
            self.cond_mlp = nn.Sequential(
                nn.Linear(global_cond_dim, n_emb), nn.SiLU(), nn.Linear(n_emb, n_emb)
            )

        self.blocks = nn.ModuleList(
            [DiTBlock(n_emb, n_head, p_drop) for _ in range(n_layer)]
        )
        self.norm_out = nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)
        self.ada_ln_out = nn.Sequential(nn.SiLU(), nn.Linear(n_emb, 2 * n_emb))
        self.head = nn.Linear(n_emb, input_dim)

        self.causal = causal
        self.max_positions = max_positions

        # init
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.zeros_(self.ada_ln_out[-1].weight)
        nn.init.zeros_(self.ada_ln_out[-1].bias)
        # zero-init the output head so the initial velocity field is ~0
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, sample, timestep, global_cond=None):
        B, T, _ = sample.shape
        assert T <= self.max_positions, (
            "action chunk length {} exceeds max_positions {}".format(T, self.max_positions)
        )

        # normalize timestep to a [B] float tensor
        t = timestep
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=sample.device)
        t = t.to(sample.device)
        if t.ndim == 0:
            t = t.expand(B)

        c = self.time_mlp(t)  # [B, n_emb]
        if self.has_cond and global_cond is not None:
            c = c + self.cond_mlp(global_cond)

        x = self.input_proj(sample) + self.pos_emb[:, :T]
        x = self.drop(x)

        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=sample.device), diagonal=1
            )

        for block in self.blocks:
            x = block(x, c, attn_mask=attn_mask)

        shift, scale = self.ada_ln_out(c).chunk(2, dim=-1)
        x = modulate(self.norm_out(x), shift, scale)
        return self.head(x)
