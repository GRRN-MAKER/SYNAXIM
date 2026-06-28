#!/usr/bin/env python3
"""
SYNAXIM Symbiotic Gate Adapter Training — Memory-Optimized
===========================================================
Trains a tiny adapter (~33K params) that teaches the Symbiotic Gate
to approximate standard attention. Runs on CPU with ~6 GB RAM.

Usage: python3 train_adapter.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc, os, time, json

# ── Config ──
MODEL_ID = "unsloth/Llama-3.2-1B"
NUM_EPOCHS = 20
PRINT_EVERY = 5
LR = 1e-2
MAX_SEQ = 16   # Short sequences to save RAM
TRAIN_LAYERS = 4  # Only train adapter for first N layers (saves RAM)

device = "cpu"

print("=" * 60)
print("  SYNAXIM — Symbiotic Gate Adapter Training")
print("  Memory-optimized for CPU (12 GB RAM)")
print("=" * 60)

# ── Step 1: Load model ──
print("\n[1/5] Loading model...")
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()

D = model.config.hidden_size
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
n_kv = model.config.num_key_value_heads
head_dim = D // n_heads

print(f"  Loaded in {time.time()-t0:.1f}s: {n_layers} layers, D={D}, {n_heads}Q/{n_kv}KV")
gc.collect()

# ── Step 2: Generate teacher data (one at a time to save RAM) ──
print("\n[2/5] Generating teacher hidden states...")

training_texts = [
    "The capital of France is Paris, which is known for",
    "In mathematics, the Pythagorean theorem states that",
    "Water boils at 100 degrees Celsius at sea level",
    "The largest planet in our solar system is Jupiter",
    "Artificial intelligence is transforming the way we",
    "Shakespeare wrote many famous plays including",
    "Photosynthesis is the process by which plants",
    "Machine learning algorithms can be classified into",
    "DNA stands for deoxyribonucleic acid and carries",
    "The theory of relativity was proposed by Albert",
    "Programming languages such as Python and Java",
    "Neural networks are inspired by the structure of",
    "Climate change is caused primarily by greenhouse",
    "Quantum computing uses qubits instead of classical",
    "The Milky Way galaxy contains hundreds of billions",
    "Gravity is one of the four fundamental forces of",
    "Hello, how are you doing today? I am doing",
    "Once upon a time, in a land far away, there",
    "The quick brown fox jumps over the lazy dog",
    "The best way to learn programming is to practice",
]

teacher_data = []
for i, text in enumerate(training_texts):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        # Only keep the hidden states we need, detached
        hiddens = [h.detach().clone() for h in outputs.hidden_states]
        del outputs
    teacher_data.append({
        "input_ids": inputs["input_ids"],
        "hidden_states": hiddens,
    })
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(training_texts)} processed")
        gc.collect()

print(f"  {len(teacher_data)} samples ready")
gc.collect()

# Verify model works
test_in = tokenizer("Hello", return_tensors="pt")
with torch.no_grad():
    test_out = model.generate(**test_in, max_new_tokens=15, do_sample=False)
print(f"  Teacher says: '{tokenizer.decode(test_out[0], skip_special_tokens=True)}'")

# ── Step 3: Define adapter ──
print("\n[3/5] Creating Symbiotic Gate Adapter...")

class SymbioticGateAdapter(nn.Module):
    def __init__(self, D, num_layers):
        super().__init__()
        self.D = D
        self.num_layers = num_layers
        self.gate_bias = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])
        self.gate_scale = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_layers)])
        self.output_scale = nn.ParameterList([nn.Parameter(torch.ones(D) * 0.1) for _ in range(num_layers)])
        self.mix_alpha = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(num_layers)])

    def symbiotic_forward(self, x, M, layer_idx, q, k, v, W_o):
        seq_len = x.shape[0]
        scale = 1.0 / (head_dim ** 0.5)

        # Gate score from Q·K similarity
        q_h = q.view(seq_len, n_heads, head_dim)
        k_h = k.view(seq_len, n_kv, head_dim)
        if n_kv < n_heads:
            k_h = k_h.repeat_interleave(n_heads // n_kv, dim=1)
        gate_raw = (q_h * k_h).sum(-1).mean(-1) * scale
        gate_input = gate_raw * self.gate_scale[layer_idx] + self.gate_bias[layer_idx]
        gate = torch.sigmoid(gate_input).unsqueeze(-1)

        # GQA-expand K,V to D dimension
        k_exp = k.view(seq_len, n_kv, head_dim)
        v_exp = v.view(seq_len, n_kv, head_dim)
        if n_kv < n_heads:
            k_exp = k_exp.repeat_interleave(n_heads // n_kv, dim=1)
            v_exp = v_exp.repeat_interleave(n_heads // n_kv, dim=1)
        k_full = k_exp.reshape(seq_len, -1)
        v_full = v_exp.reshape(seq_len, -1)

        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            k_t = k_full[t]
            v_t = v_full[t]
            q_t = q[t]
            g_t = gate[t]

            k_norm = k_t / (k_t.norm() + 1e-8)
            v_norm = v_t / (v_t.norm() + 1e-8) * x_t.norm()

            imprint = torch.outer(k_norm, v_norm)
            M = g_t * M + (1.0 - g_t) * imprint

            retrieved = q_t @ M
            out_t = retrieved @ W_o.T
            out_t = out_t * self.output_scale[layer_idx]

            alpha = torch.sigmoid(self.mix_alpha[layer_idx])
            out_t = alpha * out_t + (1.0 - alpha) * x_t
            outputs.append(out_t)

        return torch.stack(outputs), M

adapter = SymbioticGateAdapter(D, TRAIN_LAYERS)
n_params = sum(p.numel() for p in adapter.parameters())
print(f"  {n_params:,} trainable parameters (for {TRAIN_LAYERS} layers)")

# ── Step 4: Train (one layer at a time to save RAM) ──
print(f"\n[4/5] Training for {NUM_EPOCHS} epochs on {len(teacher_data)} samples ({TRAIN_LAYERS} layers)...")
print()

optimizer = optim.AdamW(adapter.parameters(), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(teacher_data))

best_loss = float('inf')
best_state = None

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    t_epoch = time.time()

    for sample in teacher_data:
        input_ids = sample["input_ids"]
        teacher_hiddens = sample["hidden_states"]
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            embeds = model.model.embed_tokens(input_ids).squeeze(0)

        h = embeds.float()
        sample_loss = torch.tensor(0.0, requires_grad=True)

        # Train one layer at a time — only one M matrix in RAM
        for layer_idx in range(TRAIN_LAYERS):
            layer = model.model.layers[layer_idx]
            teacher_h = teacher_hiddens[layer_idx + 1].squeeze(0)

            M = torch.zeros(D, D)

            with torch.no_grad():
                h_normed = layer.input_layernorm(h).float()
                q = layer.self_attn.q_proj(h_normed).float()
                k = layer.self_attn.k_proj(h_normed).float()
                v = layer.self_attn.v_proj(h_normed).float()
                W_o = layer.self_attn.o_proj.weight.float()

            attn_out, _ = adapter.symbiotic_forward(
                h_normed, M, layer_idx, q, k, v, W_o
            )

            h_new = h + attn_out

            with torch.no_grad():
                h_mlp_normed = layer.post_attention_layernorm(h_new).float()
                mlp_out = layer.mlp(h_mlp_normed).float()

            h_new = h_new + mlp_out
            layer_loss = nn.functional.mse_loss(h_new, teacher_h)
            sample_loss = sample_loss + layer_loss
            h = h_new.detach()

            del M, q, k, v, W_o, h_normed, attn_out, h_mlp_normed, mlp_out
            gc.collect()

        sample_loss = sample_loss / TRAIN_LAYERS
        optimizer.zero_grad()
        sample_loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        epoch_loss += sample_loss.item()

        del sample_loss, h, embeds, h_new
        gc.collect()

    avg_loss = epoch_loss / len(teacher_data)
    dt = time.time() - t_epoch

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

    if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
        lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | LR: {lr:.6f} | {dt:.0f}s")

adapter.load_state_dict(best_state)
print(f"\n  Training complete! Best loss: {best_loss:.6f}")

# ── Step 5: Test generation ──
print("\n[5/5] Testing adapted generation...")

def generate_adapted(prompt, max_tokens=25):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    generated = input_ids[0].tolist()
    M_mats = [torch.zeros(D, D) for _ in range(TRAIN_LAYERS)]

    all_ids = generated.copy()

    for step in range(len(generated) + max_tokens):
        tid = generated[step] if step < len(generated) else next_token

        with torch.no_grad():
            embed = model.model.embed_tokens(torch.tensor([[tid]])).squeeze()
            h = embed.float().unsqueeze(0)

            # Adapted layers (use symbiotic gate)
            for li in range(TRAIN_LAYERS):
                layer = model.model.layers[li]
                h_n = layer.input_layernorm(h).float()
                q = layer.self_attn.q_proj(h_n).float()
                k = layer.self_attn.k_proj(h_n).float()
                v = layer.self_attn.v_proj(h_n).float()
                W_o = layer.self_attn.o_proj.weight.float()
                attn_out, M_mats[li] = adapter.symbiotic_forward(h_n, M_mats[li], li, q, k, v, W_o)
                h = h + attn_out
                hm = layer.post_attention_layernorm(h).float()
                mlp_out = layer.mlp(hm).float()
                h = h + mlp_out

            # Remaining layers: use symbiotic gate too (with default params)
            # since position_embeddings are needed for standard attn
            for li in range(TRAIN_LAYERS, n_layers):
                layer = model.model.layers[li]
                h_n = layer.input_layernorm(h).float()
                q = layer.self_attn.q_proj(h_n).float()
                k = layer.self_attn.k_proj(h_n).float()
                v = layer.self_attn.v_proj(h_n).float()
                W_o = layer.self_attn.o_proj.weight.float()

                # Simple symbiotic gate (no adapter, fixed params)
                seq_len = h_n.shape[0]
                k_exp = k.view(seq_len, n_kv, head_dim)
                v_exp = v.view(seq_len, n_kv, head_dim)
                if n_kv < n_heads:
                    k_exp = k_exp.repeat_interleave(n_heads // n_kv, dim=1)
                    v_exp = v_exp.repeat_interleave(n_heads // n_kv, dim=1)
                k_full = k_exp.reshape(seq_len, -1)
                v_full = v_exp.reshape(seq_len, -1)
                q_flat = q.reshape(seq_len, -1)

                for t in range(seq_len):
                    k_t = k_full[t]
                    v_t = v_full[t]
                    k_norm = k_t / (k_t.norm() + 1e-8)
                    v_norm = v_t / (v_t.norm() + 1e-8) * h_n[t].norm()
                    # Fixed gate of 0.9 (high retention for untrained layers)
                    if not hasattr(generate_adapted, f'_M_{li}'):
                        setattr(generate_adapted, f'_M_{li}', torch.zeros(D, D))
                    M_cur = getattr(generate_adapted, f'_M_{li}')
                    M_cur = 0.9 * M_cur + 0.1 * torch.outer(k_norm, v_norm)
                    setattr(generate_adapted, f'_M_{li}', M_cur)

                attn_out = (q_flat @ getattr(generate_adapted, f'_M_{li}')) @ W_o.T
                attn_out = attn_out * 0.1  # Dampen untrained layers
                h = h + attn_out
                hm = layer.post_attention_layernorm(h).float()
                mlp_out = layer.mlp(hm).float()
                h = h + mlp_out

            h_final = model.model.norm(h).float()
            logits = (h_final @ model.lm_head.weight.float().T).squeeze()

        if step >= len(generated) - 1:
            next_token = logits.argmax().item()
            all_ids.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break

    return tokenizer.decode(all_ids, skip_special_tokens=True)

prompts = [
    "Hello, how are you",
    "The capital of France is",
    "Once upon a time",
    "Python is a programming language",
    "The meaning of life is",
]

for p in prompts:
    # Reset untrained layer M matrices between prompts
    for li in range(TRAIN_LAYERS, n_layers):
        if hasattr(generate_adapted, f'_M_{li}'):
            setattr(generate_adapted, f'_M_{li}', torch.zeros(D, D))
    result = generate_adapted(p, max_tokens=25)
    print(f"  '{p}' → '{result}'")

# ── Save adapter ──
print("\n[SAVE] Exporting adapter...")
os.makedirs("./symbiotic_adapter", exist_ok=True)
torch.save(adapter.state_dict(), "./symbiotic_adapter/symbiotic_adapter.pt")
meta = {
    "engine": "SYNAXIM", "version": "0.1.1",
    "base_model": MODEL_ID, "D": D, "n_layers": n_layers, "train_layers": TRAIN_LAYERS,
    "best_loss": best_loss, "parameters": n_params,
}
with open("./symbiotic_adapter/adapter_config.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Saved to ./symbiotic_adapter/ ({os.path.getsize('./symbiotic_adapter/symbiotic_adapter.pt')/1024:.0f} KB)")
print("\nDone!")
