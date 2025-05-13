# Use a pipeline as a high-level helper
from transformers import pipeline
from tqdm import tqdm

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
for i in tqdm(range(100)):
    messages = [
        {
            "role": "system",
            "content": "You are a paraphrasing assistant. For every sentence, Generate 20 different paraphrases of the following sentence that retain the same meaning but use different wording. Only output the paraphrases, one per line, without any additional text."
        },
        {"role": "user", "content": "I am sleeping in my bed."},
    ]
    out = pipe(messages, max_new_tokens=2048)
    print(out)