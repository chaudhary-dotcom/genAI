import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN not found")


embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",
    huggingfacehub_api_token=token,
    
)

text1 = "Nepal is a beautiful country"
text2 = "Nepal is wonderful nation"
text3 = "I like eating Icecream"

vector1 = embeddings.embed_query(text1)
vector2 = embeddings.embed_query(text2)
vector3 = embeddings.embed_query(text3)

print(cosine_similarity([vector1], [vector2])[0][0])
print(cosine_similarity([vector1], [vector3])[0][0])

print(vector1)

print("Dimensions: ", len(vector1))
