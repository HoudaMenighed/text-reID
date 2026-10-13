import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_model import LayerNorm, QuickGELU


class DescriptionConditionedGate(nn.Module):
    """Scores tokens using the token itself plus a conditioning global feature."""
    def __init__(self, token_dim=512, cond_dim=512, hidden_dim=128):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            QuickGELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens, condition):
        # tokens: [N, L, D], condition: [N, D]
        dtype = self.token_proj.weight.dtype
        tokens = tokens.to(dtype)
        condition = condition.to(dtype)
        x = self.token_proj(tokens) + self.cond_proj(condition).unsqueeze(1)
        scores = self.mlp(x).squeeze(-1)
        return scores.float()


def straight_through_topk(tokens, scores, k, mask=None, temperature=1.0):
    """Hard Top-K in the forward pass with a softmax straight-through gradient."""
    if mask is not None:
        scores_for_selection = scores.masked_fill(~mask, float('-inf'))
    else:
        scores_for_selection = scores

    k = min(k, scores_for_selection.size(1))
    _, indices = torch.topk(scores_for_selection, k=k, dim=1, largest=True, sorted=False)
    hard = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))

    finite_scores = scores_for_selection.masked_fill(~torch.isfinite(scores_for_selection), -1e4)
    weights = torch.softmax(finite_scores / temperature, dim=1)
    selected_weights = torch.gather(weights, 1, indices)

    # Forward == hard tokens. Backward receives gradients through the gate scores.
    selected = hard + hard * (selected_weights.unsqueeze(-1) - selected_weights.detach().unsqueeze(-1))
    return selected, indices


class EfficientFineGrained(nn.Module):
    """Token-gated bidirectional cross-attention and fine-grained similarity."""
    def __init__(self, embed_dim=512, gate_dim=128, heads=8, dropout=0.1,
                 visual_topk=64, text_topk=20, temperature=0.07):
        super().__init__()
        self.visual_topk = visual_topk
        self.text_topk = text_topk
        self.temperature = temperature

        self.visual_gate = DescriptionConditionedGate(embed_dim, embed_dim, gate_dim)
        self.text_gate = DescriptionConditionedGate(embed_dim, embed_dim, gate_dim)

        self.text_to_visual = nn.MultiheadAttention(
            embed_dim, heads, dropout=dropout, batch_first=True
        )
        self.visual_to_text = nn.MultiheadAttention(
            embed_dim, heads, dropout=dropout, batch_first=True
        )
        self.text_ln = LayerNorm(embed_dim)
        self.visual_ln = LayerNorm(embed_dim)
        self.text_out_ln = LayerNorm(embed_dim)
        self.visual_out_ln = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _normalize(x):
        return F.normalize(x.float(), dim=-1)

    def select_text(self, text_tokens, text_mask, global_text):
        scores = self.text_gate(text_tokens, global_text)
        return straight_through_topk(
            text_tokens, scores, self.text_topk, mask=text_mask, temperature=self.temperature
        )

    def select_visual(self, visual_tokens, global_text):
        # visual_tokens: [N, V, D], global_text: [N, D]
        scores = self.visual_gate(visual_tokens, global_text)
        return straight_through_topk(
            visual_tokens, scores, self.visual_topk, temperature=self.temperature
        )

    def pair_similarity(self, text_tokens, visual_tokens):
        """
        Compute pairwise FG similarity for already selected tokens.
        Inputs are [N, Kt, D] and [N, Kv, D], one positive/negative pair per row.
        """
        # MultiheadAttention may be converted to fp16 by IRRA's convert_weights().
        attn_dtype = self.text_to_visual.in_proj_weight.dtype
        q_t = self.text_ln(text_tokens.float()).to(attn_dtype)
        kv_v = self.visual_ln(visual_tokens.float()).to(attn_dtype)
        attended_t, _ = self.text_to_visual(q_t, kv_v, kv_v, need_weights=False)
        attended_t = self.text_out_ln((attended_t + q_t).float())

        q_v = self.visual_ln(visual_tokens.float()).to(attn_dtype)
        kv_t = self.text_ln(text_tokens.float()).to(attn_dtype)
        attended_v, _ = self.visual_to_text(q_v, kv_t, kv_t, need_weights=False)
        attended_v = self.visual_out_ln((attended_v + q_v).float())

        t = self._normalize(text_tokens)
        v = self._normalize(visual_tokens)
        at = self._normalize(attended_t)
        av = self._normalize(attended_v)

        sim_t2v = torch.matmul(at, v.transpose(1, 2)).amax(dim=-1).mean(dim=-1)
        sim_v2t = torch.matmul(av, t.transpose(1, 2)).amax(dim=-1).mean(dim=-1)
        return 0.5 * (sim_t2v + sim_v2t)

    def pairwise_logits(self, text_tokens, text_mask, global_text, visual_tokens,
                        chunk_size=8):
        """
        Build a BxB text-to-image FG similarity matrix in query chunks.
        Visual gating is description-conditioned, so each candidate image is
        scored with the current text query rather than with a fixed image gate.
        """
        B = text_tokens.size(0)
        all_logits = []

        selected_text, _ = self.select_text(text_tokens, text_mask, global_text)
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            q = end - start

            t_sel = selected_text[start:end]  # [q, Kt, D]
            t_sel = t_sel.unsqueeze(1).expand(q, B, -1, -1).reshape(q * B, t_sel.size(1), -1)

            v_raw = visual_tokens.unsqueeze(0).expand(q, B, -1, -1).reshape(q * B, visual_tokens.size(1), -1)
            t_global = global_text[start:end].unsqueeze(1).expand(q, B, -1).reshape(q * B, -1)

            v_sel, _ = self.select_visual(v_raw, t_global)
            logits = self.pair_similarity(t_sel, v_sel).view(q, B)
            all_logits.append(logits)

        return torch.cat(all_logits, dim=0)

    def positive_similarity(self, text_tokens, text_mask, global_text, visual_tokens):
        """Positive-pair FG similarity, useful for debugging/ablation."""
        t_sel, _ = self.select_text(text_tokens, text_mask, global_text)
        v_sel, _ = self.select_visual(visual_tokens, global_text)
        return self.pair_similarity(t_sel, v_sel)
