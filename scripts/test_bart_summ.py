import bart_summarizer as b_s
import pandas as pd
from nltk.tokenize import sent_tokenize
df = pd.read_csv('../SummEval/Summaries-Extractive2.csv')

articles = df['full_fragment'].tolist()

# only get sbux for now
articles = [articles[2]]
summaries = []
for article in articles:
    summary = b_s.abstractive_sent(article,max_size=6)
    summaries.append(summary)
    sent = sent_tokenize(summary)
    print(f"number of sentences: {len(sent)}")

for summary in summaries:
    print("\nThis is summary:\n")
    print(summary)

