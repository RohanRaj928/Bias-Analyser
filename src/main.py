import os

from newspaper import Article
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification
import pathlib
import torch
import nltk
import numpy as np
import tqdm
import sys
from captum.attr import LayerIntegratedGradients


base_dir = pathlib.Path(__file__).resolve().parent.parent
filepath = base_dir / 'Models' / 'Classification'

# Loading Tokenizer and model
tokenizer = LongformerTokenizerFast.from_pretrained(filepath)
model = LongformerForSequenceClassification.from_pretrained(filepath)
model.eval()

 # Get and parse article
url = input("Enter url >> ")
article = Article(url)
article.download()
article.parse()


def predict(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=4096)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    return probs


probabilities = predict(article.text)
predicted_class = torch.argmax(probabilities, dim=-1).item()
print("Predicted class:", predicted_class)
print("Probabilities:", probabilities)

### Model Explainability ###

if input("Do in depth analysis[y/N]?").upper() != "Y":
    sys.exit()


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model.to(device)


# Forward Function
def forward_func(input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits


lig = LayerIntegratedGradients(forward_func, model.longformer.embeddings.word_embeddings)

inputs = tokenizer(article.text, return_tensors='pt', padding=True, truncation=True, max_length=4096, return_offsets_mapping=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

attributions, delta = lig.attribute(
    inputs = input_ids,
    additional_forward_args=(attention_mask,),
    target=predicted_class,
    return_convergence_delta=True,
    n_steps=30,
    internal_batch_size=4
)

token_attributions = attributions.sum(dim=1).squeeze(0).detach().numpy()
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Aggregate tokens to sentence level
sentence_scores = []
idx = 0

sentences = nltk.sent_tokenize(article.text) # Get list of sentences

for sentence in sentences:
    sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
    sentence_len = len(sentence_tokens)

    if sentence_len == 0:
        continue

    sentence_score = np.mean(token_attributions[idx:idx+sentence_len])
    sentence_scores.append((sentence, sentence_score))
    idx += sentence_len

# Sort by importance
sentence_scores = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

print("\nTop contributing sentences:")
for sent, score in sentence_scores[:5]:
    direction = "↑" if score > 0 else "↓"
    print(f"[{direction} {score:.4f}] {sent}")
