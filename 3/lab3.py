import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "ai-forever/ruBert-base"
TARGET_WORDS = ["жарить", "тушить"]
TEXT = "Мясо можно [MASK] на сковороде."
TOP_K = 10

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

inputs = tokenizer(
    TEXT.replace("[MASK]", tokenizer.mask_token),
    return_tensors="pt"
).to(device)

mask_index = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()

with torch.inference_mode():
    logits = model(**inputs).logits[0, mask_index]
    probabilities = F.softmax(logits, dim=-1)
    top = torch.topk(probabilities, TOP_K)

words = [tokenizer.decode([token_id]).strip() for token_id in top.indices]

print(f"\nПредложение: {TEXT}\n")

for rank, (word, probability) in enumerate(zip(words, top.values), 1):
    print(f"{rank:<4}{word:<20}{probability.item():>13.4%}")

print("\nПроверка целевых слов:")
for word in TARGET_WORDS:
    position = words.index(word) + 1 if word in words else None
    result = f"место: {position}" if position else "не найдено"
    print(f"  {word:<10} {result}")
