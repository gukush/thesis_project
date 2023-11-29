from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import re

# Possible future direction - training the BART model to some text that we need 

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
#summarizer = pipeline("summarization", model="facebook/bart-base")

# definition of modes
# 0 - number of sentences
# 1 - number of characters
# max value takes the number value

def abstractive(text,max_size, mode=0):
    chunk_size = 1024
    num_calls = 0
    #text_ = text[:1024] # temporary workaround because model was crashing TODO: fix the tokenization
    #summary = summarizer(text_)
    #inputs = tokenizer(text, max_length=2048, return_tensors='pt', truncation=True)
    # TODO: ulepszyć preprocessing o podział na zdania i preprocessing
    # odrzucanie zdan za
    # TODO: write a function that groups sentences into list of str of at most length 1024 
    # but which does not break sentences in half
    #

    big_summary = text
    if mode == 0:
        val = len(big_summary.split('.'))
    elif mode == 1:
        val = len(big_summary)
    while val > max_size:
        # if we use facebook/bart-large-cnn model, then we can count sentences with a dot
        big_summary = re.sub(r'[^A-Za-z\s0-9\.]',' ',big_summary) 
        #summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        chunk_list = [big_summary[i:i + chunk_size] for i in range(0, len(big_summary), chunk_size)] 
        summaries = []
        for chunk in chunk_list:
            print(f"working on chunk {chunk}")
            inputs = tokenizer.encode(chunk, add_special_tokens=True)  
            # Summarize each chunk (maybe replace max_length with max_size to see performance impact)
            summary_ids = model.generate(torch.tensor([inputs]), num_beams=4, max_length=200, early_stopping=True)
            num_calls = num_calls + 1
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        # Combine summaries
        big_summary = ' '.join(summaries)
        val = len(big_summary.split('.'))
        if mode == 0:
            val = len(big_summary.split('.'))
        elif mode == 1:
            val = len(big_summary)
        print(f"length of big summary is {len(big_summary)}, call number: {num_calls}")
        print(big_summary)
    # Further summarization to shrink output
    # limit
    #while big_summary > limit:
    #    inputs = tokenizer.encode(big_summary,add_special_tokens=True)
    #    input_list = [inputs[i:i + chunk_size] for i in range(0,len(inputs),chunk_size)]
    # Decode and print summary
    #summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return big_summary


