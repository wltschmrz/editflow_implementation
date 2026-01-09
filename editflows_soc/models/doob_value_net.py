import torch
import torch.nn as nn
import torch.nn.functional as F


class DoobValueNet(nn.Module):
    """A lightweight potential network for Doob/value-difference control on discrete edit CTMCs.

    We parameterize a time-dependent additive potential:
        V(x,t) = sum_i phi_i(x_i; h_i) + psi(|x|, t)

    - phi_i(·) is produced as token logits over DNA vocab at each position i (from base hidden h_i).
    - psi(L,t) is a scalar length potential from an MLP on (t, L_norm).

    This is designed so that exp(V(y,t)-V(x,t)) gives a *structural* log-rate-add tilting:
        Q^u_t(x->y) = Q_t(x->y) * exp(V(y,t)-V(x,t))
    """

    def __init__(self, d_model: int, dna_vocab: int = 4, len_mlp_hidden: int = 128):
        super().__init__()
        self.dna_vocab = int(dna_vocab)

        # per-position token potentials: (B,L,D)->(B,L,V)
        self.token_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.dna_vocab),
        )

        # length potential psi(L,t): (B,2)->(B,1)
        self.len_mlp = nn.Sequential(
            nn.Linear(2, len_mlp_hidden),
            nn.GELU(),
            nn.Linear(len_mlp_hidden, len_mlp_hidden),
            nn.GELU(),
            nn.Linear(len_mlp_hidden, 1),
        )

    def psi_len(self, t: torch.Tensor, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """psi(|x|,t) as (B,) float."""
        t = t.float()
        lengths = lengths.float()
        # normalize length to roughly [0,1]
        L_norm = lengths / float(max(max_len, 1))
        inp = torch.stack([t, L_norm], dim=-1)  # (B,2)
        return self.len_mlp(inp).squeeze(-1)    # (B,)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return token potentials logits over DNA vocab: (B,L,V)."""
        return self.token_head(h)
