import re
from huggingface_hub import InferenceClient

with open('.ENV', encoding='utf-8') as f:
    text = f.read()
match = re.search(r'HUGGINGFACEHUB_ACCESS_TOKEN\s*=\s*"([^"]+)"', text)
if not match:
    raise SystemExit('Token not found')
token = match.group(1)

models = [
    'microsoft/Phi-3.5-mini-instruct',
    'google/gemma-2-2b-it',
    'meta-llama/Llama-3.2-1B-Instruct',
    'Qwen/Qwen2.5-0.5B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
]

for model in models:
    try:
        client = InferenceClient(model=model, token=token, provider='auto', timeout=60)
        response = client.chat_completion(
            messages=[{'role': 'user', 'content': 'Say hello in one short sentence.'}],
            max_tokens=20,
        )
        print('SUCCESS', model, response)
    except Exception as e:
        print('FAIL', model, type(e).__name__, e)
