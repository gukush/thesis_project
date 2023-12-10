from test_toc import get_toc_score, find_best_candidate
import os
import fitz
import pandas as pd


if __name__ == '__main__':
    dir = '../examples/'
    error_count_bench = 7
    x_tolerance = 17
    y_tolerance = 5
    df = pd.read_csv('../toc_pred_09_12.csv')
    count_better = 0
    count_worse = 0
    pred_old_cnt = 0
    total_cnt = 0
    for index, row in df.iterrows():
        full_path = os.path.join(dir,row['filename'])
        total_cnt = total_cnt + 1
        doc = fitz.open(full_path)
        candidate = find_best_candidate(doc,x_tolerance,y_tolerance)
        if row['page_pred'] == row['page_true']:
            pred_old_cnt = pred_old_cnt + 1
        #candidate = get_toc_score(page,doc.page_count,x_tolerance,y_tolerance)
        # adjusting for 0 indexing
        if (candidate[0]+1) == row['page_true'] and (candidate[0]+1) != row['page_pred']:
            print(f"for {row['filename']} model with changes is better than 09.12 version")
            print(f"predicted page: {candidate[0]+1}, correct: {row['page_true']}")
            count_better = count_better + 1
        elif (candidate[0]+1) != row['page_true'] and row['page_pred'] == row['page_true']:
            print(f"for {row['filename']} model with changes is worse than 09.12 version")
            print(f"predicted page: {candidate[0]+1}, correct: {row['page_true']}")
            count_worse = count_worse + 1
        else:
            print(f"for {row['filename']} model with changes is the same as 09.12 version")
            print(f"predicted page: {candidate[0]+1}, correct: {row['page_true']}")
    print("difference in performance:")
    print(f"{count_worse} times it performed worse")
    print(f"{count_better} times it performed better")
    print(f"percent of correct predictions 09.12 version {pred_old_cnt/total_cnt}")

        #print(candidate)


 



