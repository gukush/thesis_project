from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import re
from nltk.tokenize import sent_tokenize

# Possible future direction - training the BART model to some text that we need 

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
#summarizer = pipeline("summarization", model="facebook/bart-base")

# definition of modes
# 0 - number of sentences
# 1 - number of characters
# max value takes the number value

def abstractive_sent(text,max_size):
    #
    # Workflowe (mode 0): 
    # 1 tokenize big_summary into sentences
    # 2 merge sentences into at most chunk_size file lengths
    #
    g = 3
    chunk_size = 1024
    #sentence_l = 160
    num_calls = 0
    big_summary = text
    big_summary = re.sub(r'[^A-Za-z\s0-9\.\(\)\\,\;]',' ',big_summary) 
    sentences = sent_tokenize(big_summary)
    while len(sentences) > max_size:
        chunk_list = []
        chunk = ''
        # forced workaround to generate the amount of sentences needed
        if len(sentences) < max_size * 1.8 or len(sentences) < max_size + 4:
            # we extract the biggest sentences and we merge the smallest
            chunk_list = [(i,chunk) for i, chunk in enumerate(sentences)]
            sorted_chunk_list = sorted(chunk_list,key=lambda x: len(x[1]),reverse=True) #take longest
            kept_sentences = sorted_chunk_list[:max_size]
            kept_sentences = sorted(kept_sentences,key=lambda x:x[0]) # initial order they appeared in
            kept_sentences = [i[1] for i in kept_sentences]
            return ' '.join(kept_sentences)
            #if len(big_summary) < sentence_l*max_size: # then it can be chunked into max_size segments 
        #        step = len(big_summary)//max_size + 1     
        #        chunk_list = [big_summary[i:i+chunk_size] for i in range(0,len(big_summary),step)]
        elif len(sentences) < chunk_size*g: # magic number wooho, use g+1 chunks to group it
            k, m = divmod(len(sentences),g+1)
            sentences_list = (sentences[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range (g+1))
            chunk_list = [' '.join(sentences) for sentences in sentences_list]
        else:
            for sentence in sentences:
                if (len(chunk) + len(sentence)) > chunk_size:
                    chunk_list.append(chunk)
                    chunk = ''
                chunk = chunk + ' ' + sentence
        summaries = []
        total_avg = sum( map(len, chunk_list) ) / len(chunk_list)
        print(f"Average chunk length: {total_avg}")
        for chunk in chunk_list:
            print(f"call {num_calls}, working on chunk\n")
            print("-------------------------------------------------")
            print(chunk)
            inputs = tokenizer.encode(chunk, add_special_tokens=True)  
            # Summarize each chunk (maybe replace max_length with max_size to see performance impact)
            print(f"------------------------------------------------")
            summary_ids = model.generate(torch.tensor([inputs]), num_beams=4, max_length=200, early_stopping=True)
            num_calls = num_calls + 1
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        big_summary = ' '.join(summaries)
        sentences = sent_tokenize(big_summary)
        print(f"length of big summary is {len(sentences)} sentences, call number: {num_calls}")
        if len(sentences) == max_size:
            break
    return big_summary




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
    big_summary = text
    if mode == 0:
        big_summary = sent_tokenize(text)
        val = len(big_summary)
    elif mode == 1:
        val = len(big_summary)
    while val > max_size:
        big_summary = re.sub(r'[^A-Za-z\s0-9\.\(\)\\,\;]',' ',big_summary) 
        chunk_list = [big_summary[i:i + chunk_size] for i in range(0, len(big_summary), chunk_size)] 
        if mode == 0:
            if len(big_summary) < chunk_size*max_size: # then it can be chunked into max_size segments 
                step = len(big_summary)//max_size + 1     
                chunk_list = [big_summary[i:i+chunk_size] for i in range(0,len(big_summary),step)]

        # if we use facebook/bart-large-cnn model, then we can count sentences with a dot
               #summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        summaries = []
        for chunk in chunk_list:
            print(f"call {num_calls}, working on chunk\n{chunk}")
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
            print(f"length of big summary is {val} sentences, call number: {num_calls}")
        elif mode == 1:
            val = len(big_summary)
            print(f"length of big summary is {val} characters, call number: {num_calls}")
        if val == max_size:
            break
        print(big_summary)
    # Further summarization to shrink output
    # limit
    #while big_summary > limit:
    #    inputs = tokenizer.encode(big_summary,add_special_tokens=True)
    #    input_list = [inputs[i:i + chunk_size] for i in range(0,len(inputs),chunk_size)]
    # Decode and print summary
    #summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return big_summary


