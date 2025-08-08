from newspaper import Article
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch


url = "https://www.bbc.co.uk/news/articles/cx2x813jm0zo"
article = Article(url)
article.download()
article.parse()

# Load tokenizer (converts text to model input)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load pretrained BERT for Classification with 2 classes (change if needed)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Tokenize the text, convert to IDs, create attention masks
inputs = tokenizer(article.text, return_tensors="pt", truncation=True, padding=True)

outputs = model(**inputs)

# Get logits (raw scores before softmax)
logits = outputs.logits

# Convert logits to predicted class
predicted_class = torch.argmax(logits, dim=1).item()

print(f"Predicted class: {predicted_class}")
