from openai import OpenAI

client = OpenAI(
   base_url="http://127.0.0.1:52625/v1", # FastFlowLM's local API endpoint
   api_key="flm", # Dummy key (FastFlowLM doesn't require authentication)
)

resp = client.embeddings.create(
   model="embed-gemma",
   input="Hi, everyone!"
)

print(resp.data[0].embedding)

