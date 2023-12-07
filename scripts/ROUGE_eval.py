from rouge import Rouge
path
def calculate_rouge_scores(human_summaries, generated_summary):
    rouge = Rouge()
    scores = {}

    for idx, human_summary in enumerate(human_summaries):
        score = rouge.get_scores(generated_summary, human_summary)
        scores[f'Summary {idx+1}'] = {
            'ROUGE-1': score[0]['rouge-1'],
            'ROUGE-2': score[0]['rouge-2'],
            'ROUGE-L': score[0]['rouge-l']
        }

    return scores

# Example usage
human_summaries = ["Human summary 1", "Human summary 2", "Human summary 3"]
generated_summary = "Generated summary"
rouge_scores = calculate_rouge_scores(human_summaries, generated_summary)

for summary, scores in rouge_scores.items():
    print(summary)
    for score_type, score_value in scores.items():
        print(f'  {score_type}: {score_value}')
