from transformers import BartTokenizer, BartForConditionalGeneration
import torch 

# Possible future direction - training the BART model to some text that we need 

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
#summarizer = pipeline("summarization", model="facebook/bart-base")

def abstractive(text,max_size):
    chunk_size = 1024
    num_calls = 0
    #text_ = text[:1024] # temporary workaround because model was crashing TODO: fix the tokenization
    #summary = summarizer(text_)
    #inputs = tokenizer(text, max_length=2048, return_tensors='pt', truncation=True)
    big_summary = text
    while len(big_summary) > max_size:
        inputs = tokenizer.encode(big_summary, add_special_tokens=True) 
        #summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        input_list = [inputs[i:i + chunk_size] for i in range(0, len(inputs), chunk_size)]
        summaries = []
        for chunk in input_list:
            # Summarize each chunk (maybe replace max_length with max_size to see performance impact)
            summary_ids = model.generate(torch.tensor([chunk]), num_beams=4, max_length=200, early_stopping=True)
            num_calls = num_calls + 1
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        # Combine summaries
        big_summary = ' '.join(summaries) 
        print(f"length of big summary is {len(big_summary)}, call number: {num_calls}")
    # Further summarization to shrink output
    # limit
    #while big_summary > limit:
    #    inputs = tokenizer.encode(big_summary,add_special_tokens=True)
    #    input_list = [inputs[i:i + chunk_size] for i in range(0,len(inputs),chunk_size)]
    # Decode and print summary
    #summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return big_summary


