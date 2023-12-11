from rouge import Rouge
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
# Function to preprocess text: remove stopwords and apply stemming
def preprocess_text(text):
    if pd.isna(text) or text is None:
        return ""

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    words = word_tokenize(text)
    filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words and word.isalpha()]
    #filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words and word.isalpha()]
    #filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words and word.isalpha()]

    return ' '.join(filtered_words)
#import thesis_summarizer as th
def calculate_rouge_scores(df, human_summary_columns):
    rouge = Rouge()
    all_scores = {}

    #human_summary_columns = df.columns[2:-1]
    #rows = df[]
    # Iterate over each row in the DataFrame
    for row_index, row in df.iterrows():
        scores = {}
        #generated_summary = preprocess_text(row.iloc[-1])
        generated_summary = row.iloc[-1]# Last column for the generated summary

        # Iterate over the human summary columns and calculate ROUGE scores
        for idx, col in enumerate(human_summary_columns):
            #print(row_index)
            human_summary = row[col]
            #human_summary = preprocess_text(row[col])
            score = rouge.get_scores(generated_summary, human_summary)[0]
            #print(score)
            scores[f'Summary {idx + 1}'] = {
                'ROUGE-1': score['rouge-1'],
                'ROUGE-2': score['rouge-2'],
                'ROUGE-L': score['rouge-l']
            }

        all_scores[df.iloc[row_index, 0]] = scores
    return all_scores


def calculate_mean_std_rouge(all_scores):
    rouge_scores = {'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': []}

    # Collect all F-scores
    for scores in all_scores.values():
        for summary_scores in scores.values():
            rouge_scores['ROUGE-1'].append(summary_scores['ROUGE-1']['f'])
            rouge_scores['ROUGE-2'].append(summary_scores['ROUGE-2']['f'])
            rouge_scores['ROUGE-L'].append(summary_scores['ROUGE-L']['f'])

    # Calculate mean and standard deviation
    mean_std_scores = {}
    for metric, scores in rouge_scores.items():
        mean_std_scores[metric] = {
            'mean': np.mean(scores),
            'std': np.std(scores)
        }

    return mean_std_scores
# def calculate_average_f_scores(rouge_scores):
#     total_f_scores = []
#     average_scores_per_row = {}
#
#     for row, summaries in rouge_scores.items():
#         row_f_scores = []
#         for summary, metrics in summaries.items():
#             f_scores = [metrics[metric]['f'] for metric in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']]
#             avg_f_score = sum(f_scores) / len(f_scores)
#             row_f_scores.append(avg_f_score)
#
#         average_row_score = sum(row_f_scores) / len(row_f_scores)
#         average_scores_per_row[row] = average_row_score
#         total_f_scores.extend(row_f_scores)
#
#     overall_average = sum(total_f_scores) / len(total_f_scores)
#     return average_scores_per_row, overall_average

def calculate_separate_average_f_scores(rouge_scores):
    average_scores_per_row = {}

    for row, summaries in rouge_scores.items():
        total_scores = {'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': []}

        for summary, metrics in summaries.items():
            for metric in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']:
                total_scores[metric].append(metrics[metric]['f'])

        average_row_scores = {metric: sum(f_scores) / len(f_scores) for metric, f_scores in total_scores.items()}
        average_scores_per_row[row] = average_row_scores

    return average_scores_per_row

def calculate_separate_average_f_scores_with_std(rouge_scores):
    average_scores_per_row = {}

    for row, summaries in rouge_scores.items():
        total_scores = {'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': []}

        for summary, metrics in summaries.items():
            for metric in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']:
                total_scores[metric].append(metrics[metric]['f'])

        # Calculate average and standard deviation for each metric
        average_row_scores = {}
        for metric, f_scores in total_scores.items():
            average_row_scores[metric] = {
                'average': np.mean(f_scores),
                'std_dev': np.std(f_scores)
            }

        average_scores_per_row[row] = average_row_scores
    return average_scores_per_row

path = "C:/Users/Admin/PycharmProjects/thesis_project/examples/Summaries_abstractive.xlsx"
df = pd.read_excel(path, engine='openpyxl')
print(df)
human_summary_columns = ['Person 1', 'Person 2', 'Person 3']
#df_without_stopword = th.RemoveStopWords(df)
rouge_scores = calculate_rouge_scores(df, human_summary_columns)
print(rouge_scores)

average_scores_per_row = calculate_separate_average_f_scores(rouge_scores)
print(average_scores_per_row)
average_f = pd.DataFrame(average_scores_per_row)
output_file_path = "C:/Users/Admin/PycharmProjects/thesis_project/examples/rouge_abstractice_scores_.xlsx"
average_f.to_excel(output_file_path, index_label='Text')





