import pandas as pd
import csv
import os
# Import all metric classes
from summ_eval.bert_score_metric import BertScoreMetric
from summ_eval.blanc_metric import BlancMetric
from summ_eval.bleu_metric import BleuMetric
from summ_eval.chrfpp_metric import ChrfppMetric
from summ_eval.cider_metric import CiderMetric
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.mover_score_metric import MoverScoreMetric
from summ_eval.rouge_metric import RougeMetric
from summ_eval.rouge_we_metric import RougeWeMetric
#from summ_eval.s3_metric import S3Metric
from summ_eval.sentence_movers_metric import SentenceMoversMetric
#from summ_eval.summa_qa_metric import SummaQAMetric
from summ_eval.supert_metric import SupertMetric
#from summ_eval.syntactic_metric import SyntacticMetric #nie

# Function to create a metric instance from its class name
def get_metric_instance(metric_class_name):
    try:
        # Assuming all metric classes have a similar constructor
        return metric_class_name()
    except Exception as e:
        print(f"Error initializing {metric_class_name.__name__}: {e}")
        return None

# List of all metric classes
metrics_classes = [
    BertScoreMetric,
    #BlancMetric,
    BleuMetric,
    ChrfppMetric,
    CiderMetric,
    RougeMetric,
    RougeWeMetric,
    MeteorMetric,
    MoverScoreMetric,
    #S3Metric,
    SentenceMoversMetric,
    #SummaQAMetric,
    SupertMetric,
    #SyntacticMetric
]

# Read data
df = pd.read_csv("Summaries-Extractive.csv")
summaries = df['Text Rank'].tolist()
multi_references = [row.tolist() for _, row in df[['Person 1', 'Person 2', 'Person 3']].iterrows()]

# ID 
# 0 - total
# 1 - Alphabet
# 2 - BVB
# 3 - Sbux
# 4 - Nvidia
# 5 - Intel
# 6 - PZU

# Iterate over metrics
for metric_class in metrics_classes:
    metric = get_metric_instance(metric_class)
    if metric is None:
        continue
    if os.path.isfile(f"{metric_class.__name__}_results.csv"):
        continue
    print(f"Evaluating with {metric_class.__name__}")
    try:
        if metric.supports_multi_ref:
            # Evaluate batch
            out_list = []
            out = metric.evaluate_batch(summaries, multi_references)
            out['ID'] = 0
            print(out)
            out_list.append(out)
            # Evaluate each example
            id = 0
            for summary, references in zip(summaries, multi_references):
                id = id + 1
                new_out = metric.evaluate_batch([summary], [references])
                new_out['ID'] = id
                out_list.append(new_out)
            with open(f"{metric_class.__name__}_results.csv",'w+',encoding='utf-8',newline='') as out_file:
                fc = csv.DictWriter(out_file, 
                        fieldnames=out_list[0].keys(),

                       )
                fc.writeheader()
                fc.writerows(out_list)
            print(out_list)

        else:
            print(f"{metric_class.__name__} does not support multi-reference evaluation.")
            out_list = []
            id = 0
            for summary, references in zip(summaries, multi_references):
                id = id + 1
                new_out = metric.evaluate_batch(summary, references)
                new_out['ID'] = id
                out_list.append(new_out)
            with open(f"{metric_class.__name__}_results.csv",'w+',encoding='utf-8',newline='') as out_file:
                fc = csv.DictWriter(out_file, 
                        fieldnames=out_list[0].keys(),

                       )
                fc.writeheader()
                fc.writerows(out_list)


    except Exception as e:
        print(f"Error evaluating with {metric_class.__name__}: {e}")
