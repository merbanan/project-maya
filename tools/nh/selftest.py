"""tools/nh/selftest.py - tools/nh/ref.py's forward against transformers' NemotronH on a small model of random weights
(the torch path of modeling_nemotron_h.py; time_step_min ~0 so its dt clamp - not the model's kernels' - does not act,
and one chunk for the whole sequence: its chunked scan across chunks does not agree with itself at another chunk size,
while one chunk is the recurrence's sums).

    python tools/nh/selftest.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ref  # noqa: E402
from transformers import NemotronHConfig, NemotronHForCausalLM  # noqa: E402


class FakeST:
    """ref.py's view of the weights (st.has / st.f32 / st.t) over a state dict of float tensors."""

    def __init__(self, sd: dict):
        self.w = {}
        for k, v in sd.items():
            k = k.replace("model.", ref.PFX, 1) if k.startswith("model.") else "language_model." + k
            a = v.detach().float().numpy()
            if k.endswith("experts.up_proj") or k.endswith("experts.down_proj"):   # [E, out, in] -> one a expert
                base, role = k.rsplit(".", 1)
                for e in range(a.shape[0]):
                    self.w["%s.%d.%s.weight" % (base, e, role)] = a[e]
                continue
            self.w[k] = a
        self.t = {k: (None, "F32", v.shape, 0, 0) for k, v in self.w.items()}

    def has(self, n):
        return n in self.w

    def f32(self, n):
        return self.w[n]


def main():
    torch.manual_seed(0)
    types = ["mamba", "moe", "attention", "moe", "mamba", "moe"]
    cfg = NemotronHConfig(vocab_size=512, hidden_size=256, layers_block_type=types, num_hidden_layers=len(types),
                          mamba_num_heads=16, mamba_head_dim=32, ssm_state_size=32, n_groups=4, conv_kernel=4,
                          chunk_size=64, num_attention_heads=8, num_key_value_heads=2, head_dim=32,
                          n_routed_experts=16, num_experts_per_tok=4, moe_intermediate_size=64,
                          moe_shared_expert_intermediate_size=96, moe_latent_size=128, routed_scaling_factor=5.0,
                          n_group=1, topk_group=1, norm_topk_prob=True, mlp_hidden_act="relu2",
                          use_mamba_kernels=False, time_step_min=1e-12, layer_norm_epsilon=1e-5,
                          tie_word_embeddings=False, use_conv_bias=True, use_bias=False, mlp_bias=False)
    m = NemotronHForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif n.endswith("A_log"):
                p.copy_(torch.log(torch.rand_like(p) * 4 + 0.5))
            elif n.endswith("dt_bias"):
                p.copy_(torch.randn_like(p) * 0.5 - 1.0)
            else:
                p.copy_(torch.randn_like(p) * 0.05)
        for n, b in m.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(torch.randn_like(b) * 0.1)
    sd = dict(m.state_dict())
    st = FakeST(sd)
    c = cfg.to_dict()
    c["layers_block_type"] = types
    ids = torch.randint(0, 512, (40,)).tolist()
    with torch.no_grad():
        hf = m(torch.tensor([ids])).logits[0].float()
        out = m(torch.tensor([ids]), output_hidden_states=True)
        mine, hs = ref.forward(st, c, ids, "cpu", keep=len(ids), want_hs=True, verbose=False)
        for li, (a, b) in enumerate(zip(out.hidden_states[1:-1], hs)):
            print("layer %d %-9s max |h diff| %.3g" % (li, types[li], (a[0] - torch.from_numpy(b)).abs().max().item()))
    d = (hf - mine).abs().max().item()
    print("max |logit diff| %.3g (logits up to %.3g); argmax agree %d / %d" %
          (d, hf.abs().max().item(), (hf.argmax(-1) == mine.argmax(-1)).sum().item(), len(ids)))
    sys.exit(0 if d < 1e-3 * max(1.0, hf.abs().max().item()) else 1)


if __name__ == "__main__":
    main()
