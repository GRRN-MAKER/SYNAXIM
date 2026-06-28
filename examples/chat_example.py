"""
SYNAXIM: Chat Example
======================
Multi-turn chat with streaming output.
"""

from grrn_inference import GRRNModel

model = GRRNModel.from_pretrained("./llama-8b-symb")

# ── Simple Chat ──
result = model.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
])
print("Assistant:", result.choices[0].message["content"])
print(f"({result.usage['completion_tokens']} tokens, {result.generation_time_ms}ms)")

# ── Multi-Turn Chat ──
messages = [
    {"role": "system", "content": "You are a coding tutor."},
    {"role": "user", "content": "How do I reverse a string in Python?"},
]

result = model.chat(messages, max_tokens=200, temperature=0.7)
print("\nTutor:", result.choices[0].message["content"])

# Continue the conversation
messages.append({"role": "assistant", "content": result.choices[0].message["content"]})
messages.append({"role": "user", "content": "Now show me how to do it in Rust."})

result = model.chat(messages, max_tokens=200, temperature=0.7)
print("\nTutor:", result.choices[0].message["content"])

# ── Streaming ──
print("\n--- Streaming Example ---")
print("Story: ", end="")
for chunk in model.stream("Once upon a time in a land far away,", max_tokens=100):
    print(chunk.text, end="", flush=True)
print("\n--- Done ---")
