"""
SYNAXIM Quickstart
===================
Convert a HuggingFace model and run inference in 5 lines.
"""

from grrn_inference import GRRNModel, SymbioticConverter

# Step 1: Convert a model (only needed once)
converter = SymbioticConverter()
converter.convert(
    source="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="./llama-8b-symb",
    quantize="int4",
)

# Step 2: Load and generate
model = GRRNModel.from_pretrained("./llama-8b-symb")

result = model.generate(
    prompt="Explain quantum computing in simple terms:",
    max_tokens=200,
    temperature=0.7,
)

print(result.text)
print(f"\nTokens/sec: {result.tokens_per_second}")
print(f"Total tokens: {result.total_tokens}")
