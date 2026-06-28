"""
SYNAXIM: OpenAI-Compatible API Server
======================================
Start a server that any OpenAI client can connect to.

After running this script:
    curl http://localhost:8000/v1/chat/completions \
      -H "Authorization: Bearer my-secret-key" \
      -H "Content-Type: application/json" \
      -d '{"model":"grrn","messages":[{"role":"user","content":"Hello!"}]}'

Or use the OpenAI Python client:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="my-secret-key")
    response = client.chat.completions.create(
        model="grrn",
        messages=[{"role": "user", "content": "Hello!"}]
    )
"""

from grrn_inference import GRRNModel
from grrn_inference.server import GRRNServer

# Option 1: Quick serve
# from grrn_inference import serve
# serve("./llama-8b-symb", port=8000, api_key="my-secret-key")

# Option 2: Customize server
server = GRRNServer(
    model_path="./llama-8b-symb",
    api_key="my-secret-key",
    model_name="llama-8b-grrn",
)

# Start serving
server.run(host="0.0.0.0", port=8000)
