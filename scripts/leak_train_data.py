import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import re
from nltk.tokenize import sent_tokenize


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')



starting_str = 'The Starbucks brand is recognized throughout most of the world, and we have received high ratings in global brand value studies. To be successful in the future, particularly outside of the U.S. where the Starbucks brand and our other brands are less well-known, we believe we must preserve, grow and leverage the value of our brands across all sales channels. Brand value is based in part on consumer perceptions on a variety of subjective qualities.'
reverse_summary = starting_str
sentences = sent_tokenize(reverse_summary)
num_calls = 0
num_sentences = len(sentences)
while num_sentences < 24:
    print(f"current reverse summary has {num_sentences} sentences.")
    sentences = sent_tokenize(reverse_summary)
    summaries = []
    for sentence in sentences:
        inputs = tokenizer.encode(sentence, add_special_tokens=True)
        summary_ids = model.generate(torch.tensor([inputs]), num_beams=4, min_length=128, early_stopping=True)
        num_calls = num_calls + 1
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    reverse_summary = ' '.join(summaries)
    print("---------------------------")
    print(reverse_summary)
    print("---------------------------")
    num_sentences = len(sent_tokenize(reverse_summary))
    print(f"num sentences after reverse summarization {num_sentences}")


with open("reverse_summary.txt","w+") as f:
    f.write(reverse_summary)


