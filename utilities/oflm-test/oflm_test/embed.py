from openai import OpenAI

client = OpenAI(
   base_url="http://127.0.0.1:52625/v1", # OpenFlowLM's local API endpoint
   api_key="oflm", # Dummy key (OpenFlowLM doesn't require authentication)
)

resp = client.embeddings.create(
   model="embed-gemma",
   input="Hi, everyone!"
)

print(resp.data[0].embedding)

