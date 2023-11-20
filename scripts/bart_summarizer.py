from transformers import pipeline
import torch 

summarizer = pipeline("summarization", model="facebook/bart-base")

def abstractive(text):
    text_ = text[:1024] # temporary workaround because model was crashing
    summary = summarizer(text_)
    return summary


