# R-News 

A Python tool that analyses news articles for potential political bias 
indicators. The goal is to help readers critically evaluate news sources 
by highlighting patterns in tone and word usage.

## Installation
```
git clone https://github.com/your-username/bias-analyser.git
cd bias-analyser
pip install -r requirements.txt
```

## Usage
/ / /

## Training
python scripts used for training are also included

## Methodology
1. Preprocessing  
Articles are collected from online sources.
2. Representation  
Articles are encoded using Longformer to capture context over long sequences.
3. Bias Indicators  
Articles are classified as Left, Center or Right.
4. Output  
The final classification is given. The sentences contributed most to the model's
decision are highlighted.

## Tech Stack
* Longformer (Huggingface)
* PyTorch for training/inference
* Newspaper3k for article scraping
