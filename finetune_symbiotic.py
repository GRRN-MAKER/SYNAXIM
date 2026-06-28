import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def cell_1():
    def _():
        import marimo as mo
        return mo.md("""
        # 🧬 SYNAXIM — Symbiotic Gate Fine-Tuning

        **Goal**: Fine-tune Llama 3.2 1B so it produces coherent text through the Symbiotic Gate.

        **Method**: Knowledge distillation — use the standard attention outputs as teacher, 
        train the model to produce similar outputs through the O(1) M-matrix gate.

        **Steps**:
        1. Load Llama 3.2 1B (standard PyTorch)
        2. Generate teacher hidden states with standard attention
        3. Train a gate adapter to make Symbiotic Gate match teacher
        4. Export adapted weights to `.symb` format
        5. Push to HuggingFace
        """)


    _()
    return


@app.cell
def cell_2():
    import subprocess, sys

    # Install dependencies if needed
    deps = ["torch", "transformers", "datasets", "safetensors", "tqdm", "numpy", "huggingface-hub"]
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])
            print(f"Installed {dep}")

    import torch
    import numpy as np
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU — training will be slow but works")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    return (torch,)


@app.cell
def _():
    def _():
        import sys
        print(sys.executable)
        import torch
        return print(torch.__version__)


    _()
    return


@app.cell
def cell_3(torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_ID = "unsloth/Llama-3.2-1B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map=device
    )
    model.eval()

    D = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    head_dim = D // n_heads

    print(f"Loaded: {n_layers} layers, D={D}, {n_heads}Q/{n_kv}KV, head_dim={head_dim}")
    print(f"Device: {next(model.parameters()).device}")
    return D, device, head_dim, model, n_heads, n_kv, n_layers, tokenizer


@app.cell
def cell_4(device, model, tokenizer, torch):
    """Generate training data: input sequences + teacher hidden states."""

    # Training prompts — diverse short texts
    training_texts = [
        "The capital of France is Paris, which is known for",
        "In mathematics, the Pythagorean theorem states that",
        "Water boils at 100 degrees Celsius at sea level",
        "The largest planet in our solar system is Jupiter",
        "Artificial intelligence is transforming the way we",
        "The speed of light in vacuum is approximately",
        "Shakespeare wrote many famous plays including",
        "Photosynthesis is the process by which plants",
        "The human brain contains approximately 86 billion",
        "Machine learning algorithms can be classified into",
        "The Great Wall of China was built over many",
        "DNA stands for deoxyribonucleic acid and carries",
        "The theory of relativity was proposed by Albert",
        "Ocean currents play a vital role in regulating",
        "Programming languages such as Python and Java",
        "The periodic table organizes chemical elements by",
        "Neural networks are inspired by the structure of",
        "Climate change is caused primarily by greenhouse",
        "The Internet was originally developed as a military",
        "Quantum computing uses qubits instead of classical",
        "Evolution by natural selection was described by",
        "The Milky Way galaxy contains hundreds of billions",
        "Electricity flows through conductors due to the",
        "The Renaissance was a period of cultural rebirth",
        "Gravity is one of the four fundamental forces of",
        "Antibiotics work by killing or inhibiting the growth",
        "The Amazon rainforest produces approximately twenty",
        "Semiconductors are materials with electrical properties",
        "Black holes are regions of spacetime where gravity",
        "The United Nations was established in 1945 to",
        "Cryptography is the practice of securing communication",
        "Volcanoes form when magma from the Earth mantle",
        "Hello, how are you doing today? I am doing",
        "Once upon a time, in a land far away, there",
        "The quick brown fox jumps over the lazy dog",
        "To be or not to be, that is the question",
        "I think therefore I am, as Descartes famously",
        "In the beginning, there was nothing but darkness",
        "The best way to learn programming is to practice",
        "Yesterday I went to the store and bought some",
    ]

    print(f"Generating teacher hidden states for {len(training_texts)} sequences...")

    teacher_data = []

    for i, text in enumerate(training_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )
            # Get hidden states from each layer (list of [1, seq_len, D])
            hidden_states = [h.cpu().float() for h in outputs.hidden_states]
            logits = outputs.logits.cpu().float()

        teacher_data.append({
            "input_ids": inputs["input_ids"].cpu(),
            "hidden_states": hidden_states,  # n_layers+1 tensors
            "logits": logits,
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(training_texts)}")

    print(f"Teacher data generated: {len(teacher_data)} samples")

    # Verify the model works normally
    test_input = tokenizer("Hello, how are you", return_tensors="pt").to(device)
    with torch.no_grad():
        test_out = model.generate(**test_input, max_new_tokens=20, do_sample=False)
    print(f"\nTeacher output (standard attention):")
    print(f"  '{tokenizer.decode(test_out[0], skip_special_tokens=True)}'")
    return (teacher_data,)


@app.cell
def cell_5(D, device, head_dim, n_heads, n_kv, n_layers, torch):
    """Define the Symbiotic Gate Adapter — trainable gate parameters."""
    import torch.nn as nn

    class SymbioticGateAdapter(nn.Module):
        """
        Learns gate biases and scaling that make the Symbiotic M-matrix
        produce hidden states close to standard attention.

        For each layer, we learn:
        - gate_bias: shifts the sigmoid gate (controls retention vs imprint)
        - gate_scale: scales the gate input
        - output_scale: scales the M-matrix output
        - mix_alpha: blends M-matrix output with residual
        """
        def __init__(self, D, n_layers):
            super().__init__()
            self.D = D
            self.n_layers = n_layers

            # Per-layer learnable parameters
            self.gate_bias = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in range(n_layers)
            ])
            self.gate_scale = nn.ParameterList([
                nn.Parameter(torch.ones(1)) for _ in range(n_layers)
            ])
            self.output_scale = nn.ParameterList([
                nn.Parameter(torch.ones(D) * 0.1) for _ in range(n_layers)
            ])
            self.mix_alpha = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5)) for _ in range(n_layers)
            ])

        def symbiotic_forward(self, x, M, layer_idx, q, k, v, W_o):
            """
            Run the Symbiotic Gate with learned adaptation.

            Args:
                x: input hidden state (seq_len, D)
                M: persistent memory matrix (D, D)
                layer_idx: which layer
                q, k, v: projected Q, K, V from model's attention weights
                W_o: output projection weight

            Returns:
                output: (seq_len, D)
                M_updated: updated memory matrix
            """
            seq_len = x.shape[0]

            # Compute gate score from Q·K similarity
            scale = 1.0 / (head_dim ** 0.5)

            # Reshape for per-head computation
            q_h = q.view(seq_len, n_heads, head_dim)
            k_h = k.view(seq_len, n_kv, head_dim)

            # GQA repeat
            if n_kv < n_heads:
                k_h = k_h.repeat_interleave(n_heads // n_kv, dim=1)

            # Per-head gate scores
            gate_raw = (q_h * k_h).sum(-1).mean(-1) * scale  # (seq_len,)

            # Apply learned gate adaptation
            gate_input = gate_raw * self.gate_scale[layer_idx] + self.gate_bias[layer_idx]
            gate = torch.sigmoid(gate_input).unsqueeze(-1)  # (seq_len, 1)

            # GQA-expand K and V to full D dimension for M matrix ops
            k_exp = k.view(seq_len, n_kv, head_dim)
            v_exp = v.view(seq_len, n_kv, head_dim)
            if n_kv < n_heads:
                k_exp = k_exp.repeat_interleave(n_heads // n_kv, dim=1)
                v_exp = v_exp.repeat_interleave(n_heads // n_kv, dim=1)
            k_full = k_exp.reshape(seq_len, -1)  # (seq_len, D)
            v_full = v_exp.reshape(seq_len, -1)  # (seq_len, D)

            # Process each token through M matrix
            outputs = []
            for t in range(seq_len):
                x_t = x[t]  # (D,)
                k_t = k_full[t]  # (D,)
                v_t = v_full[t]  # (D,)
                q_t = q[t]  # (D,)
                g_t = gate[t]  # (1,)

                # Normalize key and value for stable imprints
                k_norm = k_t / (k_t.norm() + 1e-8)
                v_norm = v_t / (v_t.norm() + 1e-8) * x_t.norm()

                # Imprint: (D, D) outer product
                imprint = torch.outer(k_norm, v_norm)

                # Gated update
                M = g_t * M + (1.0 - g_t) * imprint

                # Retrieve: query through memory
                retrieved = q_t @ M

                # Output projection
                out_t = retrieved @ W_o.T

                # Apply learned output scaling
                out_t = out_t * self.output_scale[layer_idx]

                # Mix with residual (learned blending)
                alpha = torch.sigmoid(self.mix_alpha[layer_idx])
                out_t = alpha * out_t + (1.0 - alpha) * x_t

                outputs.append(out_t)

            return torch.stack(outputs), M

    adapter = SymbioticGateAdapter(D, n_layers).to(device)
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"Symbiotic Gate Adapter: {n_params:,} trainable parameters")
    print(f"  gate_bias:    {n_layers} × 1")
    print(f"  gate_scale:   {n_layers} × 1")
    print(f"  output_scale: {n_layers} × {D}")
    print(f"  mix_alpha:    {n_layers} × 1")
    return (adapter,)


@app.cell
def cell_6(D, adapter, device, model, n_layers, teacher_data, torch):
    """Train the adapter to match teacher hidden states."""
    import torch.optim as optim

    optimizer = optim.AdamW(adapter.parameters(), lr=1e-2, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    NUM_EPOCHS = 30
    PRINT_EVERY = 5

    print(f"Training adapter for {NUM_EPOCHS} epochs on {len(teacher_data)} samples...")
    print()

    best_loss = float('inf')
    best_state = None

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0

        for sample in teacher_data:
            input_ids = sample["input_ids"].to(device)
            teacher_hiddens = sample["hidden_states"]
            teacher_logits = sample["logits"].to(device)

            seq_len = input_ids.shape[1]

            with torch.no_grad():
                embeds = model.model.embed_tokens(input_ids).squeeze(0)

            h = embeds.float()
            M_matrices = [torch.zeros(D, D, device=device) for _ in range(n_layers)]

            loss = torch.tensor(0.0, device=device, requires_grad=True)

            for layer_idx in range(n_layers):
                layer = model.model.layers[layer_idx]
                teacher_h = teacher_hiddens[layer_idx + 1].squeeze(0).to(device)

                with torch.no_grad():
                    h_normed = layer.input_layernorm(h).float()
                    q = layer.self_attn.q_proj(h_normed).float()
                    k = layer.self_attn.k_proj(h_normed).float()
                    v = layer.self_attn.v_proj(h_normed).float()
                    W_o = layer.self_attn.o_proj.weight.float()

                attn_out, M_matrices[layer_idx] = adapter.symbiotic_forward(
                    h_normed, M_matrices[layer_idx], layer_idx, q, k, v, W_o
                )

                h_new = h + attn_out

                with torch.no_grad():
                    h_mlp_normed = layer.post_attention_layernorm(h_new).float()
                    mlp_out = layer.mlp(h_mlp_normed).float()

                h_new = h_new + mlp_out

                layer_loss = torch.nn.functional.mse_loss(h_new, teacher_h)
                loss = loss + layer_loss

                h = h_new.detach()

            loss = loss / n_layers

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(teacher_data)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | LR: {lr:.6f}")

    adapter.load_state_dict(best_state)
    print(f"\nTraining complete! Best loss: {best_loss:.6f}")
    return (best_loss,)


@app.cell
def cell_7(D, adapter, device, model, n_layers, next_token, tokenizer, torch):
    """Test the adapted model — generate text through the Symbiotic Gate."""

    def generate_with_symbiotic(prompt, max_tokens=50, temperature=0.7):
        """Generate text using the trained Symbiotic Gate adapter."""
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        generated = input_ids[0].tolist()
        M_matrices = [torch.zeros(D, D, device=device) for _ in range(n_layers)]

        # Process prompt + generate
        all_ids = generated.copy()

        for step in range(len(generated) + max_tokens):
            if step < len(generated):
                token_id = generated[step]
            else:
                token_id = next_token

            # Embed
            with torch.no_grad():
                embed = model.model.embed_tokens(torch.tensor([[token_id]], device=device)).squeeze()

            h = embed.float().unsqueeze(0)  # (1, D)

            for layer_idx in range(n_layers):
                layer = model.model.layers[layer_idx]

                with torch.no_grad():
                    h_normed = layer.input_layernorm(h).float()
                    q = layer.self_attn.q_proj(h_normed).float()
                    k = layer.self_attn.k_proj(h_normed).float()
                    v = layer.self_attn.v_proj(h_normed).float()
                    W_o = layer.self_attn.o_proj.weight.float()

                with torch.no_grad():
                    attn_out, M_matrices[layer_idx] = adapter.symbiotic_forward(
                        h_normed, M_matrices[layer_idx], layer_idx, q, k, v, W_o
                    )

                h = h + attn_out

                with torch.no_grad():
                    h_mlp = layer.post_attention_layernorm(h).float()
                    mlp_out = layer.mlp(h_mlp).float()
                h = h + mlp_out

            # Final norm + logits
            with torch.no_grad():
                h_final = model.model.norm(h).float()
                logits = (h_final @ model.lm_head.weight.float().T).squeeze()

            if step >= len(generated) - 1:
                # Sample
                if temperature <= 0:
                    next_token = logits.argmax().item()
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                all_ids.append(next_token)

                if next_token == tokenizer.eos_token_id:
                    break

        return tokenizer.decode(all_ids, skip_special_tokens=True)

    print("=" * 60)
    print("  SYNAXIM Symbiotic Gate — Adapted Llama 3.2 1B")
    print("=" * 60)
    print()

    test_prompts = [
        "Hello, how are you",
        "The capital of France is",
        "Once upon a time",
        "The meaning of life is",
        "Python is a programming language that",
    ]

    for prompt in test_prompts:
        result = generate_with_symbiotic(prompt, max_tokens=30, temperature=0.0)
        print(f"Prompt: '{prompt}'")
        print(f"Output: '{result}'")
        print()
    return


@app.cell
def cell_8(D, adapter, best_loss, n_layers, torch):
    """Export the adapter weights for SYNAXIM engine."""
    import os, json

    EXPORT_DIR = "./symbiotic_adapter"
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Save adapter state dict
    adapter_path = os.path.join(EXPORT_DIR, "symbiotic_adapter.pt")
    torch.save(adapter.state_dict(), adapter_path)

    # Save metadata
    meta = {
        "engine": "SYNAXIM",
        "version": "0.1.1",
        "base_model": "unsloth/Llama-3.2-1B",
        "adapter_type": "symbiotic_gate",
        "D": D,
        "n_layers": n_layers,
        "best_loss": best_loss,
        "parameters": sum(p.numel() for p in adapter.parameters()),
        "description": "Symbiotic Gate adapter trained to produce coherent output from Llama 3.2 1B through O(1) M-matrix inference"
    }
    with open(os.path.join(EXPORT_DIR, "adapter_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    file_size = os.path.getsize(adapter_path)
    print(f"Adapter saved: {adapter_path} ({file_size/1024:.1f} KB)")
    print(f"Config saved:  {EXPORT_DIR}/adapter_config.json")
    print(f"Best loss:     {best_loss:.6f}")
    print(f"Parameters:    {meta['parameters']:,}")
    return


@app.cell
def cell_9():
    def _():
        import marimo as mo
        return mo.md("""
        ## ✅ Training Complete!

        The Symbiotic Gate adapter has been trained and exported.

        **Next steps:**
        1. The adapter weights are in `./symbiotic_adapter/`
        2. These will be integrated into the SYNAXIM `.symb` format
        3. Then pushed to `GRRNNOB/SYNAXIM` on HuggingFace

        The adapter teaches the O(1) M-matrix to approximate standard attention
        outputs, so converted models produce coherent text through SYNAXIM.
        """)


    _()
    return


if __name__ == "__main__":
    app.run()
