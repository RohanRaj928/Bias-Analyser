from newspaper import Article
import torch


url = "https://www.bbc.co.uk/news/articles/cx2x813jm0zo"
article = Article(url)
article.download()
article.parse()

