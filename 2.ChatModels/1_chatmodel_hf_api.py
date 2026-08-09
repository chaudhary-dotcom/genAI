import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN not found")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    provider="featherless-ai",
    huggingfacehub_api_token=token,
    max_new_tokens=100,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke(
    "mero Naam K ho ?"
)

print(result.content)

