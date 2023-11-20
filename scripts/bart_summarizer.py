from transformers import pipeline
import torch 

# Possible future direction - training the BART model to some text that we need 


summarizer = pipeline("summarization", model="facebook/bart-base")

def abstractive(text):
    text_ = text[:1024] # temporary workaround because model was crashing TODO: fix the tokenization
    summary = summarizer(text_)
    return summary


