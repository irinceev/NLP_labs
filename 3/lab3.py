import torch
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "ai-forever/ruBert-base"
CANDIDATES = ["жарить", "тушить"]
SENTENCE = "Мясо можно [MASK] на сковороде."
N_BEST = 10

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForMaskedLM.from_pretrained(CHECKPOINT).to(device)
model.eval()

encoded = tokenizer(
    SENTENCE.replace("[MASK]", tokenizer.mask_token),
    return_tensors="pt"
).to(device)

pos = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()

with torch.inference_mode():
    scores = model(**encoded).logits[0, pos]
    dist = F.softmax(scores, dim=-1)
    best = torch.topk(dist, N_BEST)

tokens = [tokenizer.decode([tid]).strip() for tid in best.indices]

print(f"\nПредложение: {SENTENCE}\n")
for idx, (token, prob) in enumerate(zip(tokens, best.values), 1):
    print(f"{idx:<4}{token:<20}{prob.item():>13.4%}")

print("\nПроверка слов:")
for target in CANDIDATES:
    rank = tokens.index(target) + 1 if target in tokens else None
    verdict = f"место: {rank}" if rank else "не найдено"
    print(f"  {target:<10} {verdict}")
