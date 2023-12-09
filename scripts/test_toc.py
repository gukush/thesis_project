import fitz
import random
import os
import time
import pandas
import matplotlib.pyplot as plt
import string
import argparse
import csv
import io
import re

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_columns(page, x_tolerance=10,y_tolerance=5):
    blocks = page.get_text("dict")["blocks"]
    
    # Extracting numbers and their positions
    numbers_with_pos = []
    for b in blocks:
        if "lines" in b:  # Check if 'lines' key exists
            for l in b["lines"]:
                for s in l["spans"]:
                    if is_number(s["text"].strip()):
                        numbers_with_pos.append((float(s["text"]), s["bbox"]))

    # Group numbers by x-coordinate
    grouped_numbers = {}
    for num, pos in numbers_with_pos:
        x_coord = pos[0]
        found_group = False
        for x in grouped_numbers:
            if abs(x - x_coord) <= x_tolerance:
                grouped_numbers[x].append((num, pos))
                found_group = True
                break
        if not found_group:
            grouped_numbers[x_coord] = [(num, pos)]

    # Process each group
    results = []
    for x, nums in grouped_numbers.items():
        if len(nums) > 1:
            # Sorting numbers based on vertical position
            nums.sort(key=lambda x: x[1][1])

            # Extracting sorted numbers and checking if they are in increasing order
            sorted_numbers = [num for num, _ in nums]
            is_increasing = all(sorted_numbers[i] <= sorted_numbers[i + 1] for i in range(len(sorted_numbers) - 1))

            # Extracting text in the same horizontal row
            extracted_text = []
            for _, num_pos in nums:
                texts = []
                for b in blocks:
                    if "lines" in b:
                        for l in b["lines"]:
                            for s in l["spans"]:
                                if num_pos[1]-y_tolerance <= s["bbox"][1] <= num_pos[3]+y_tolerance and num_pos[0] != s["bbox"][0]:
                                    texts.append(s["text"].strip())
                extracted_text.append(texts)
            results.append((x, sorted_numbers, is_increasing, extracted_text))

    return results

def find_best_candidate(doc, x_tolerance=10, y_tolerance=5):
    best_score = 0
    best_candidate = None
    keywords = {'table':5, 'content':10, 'table of content':10}
    for page in doc:
        if page.number > doc.page_count/2: # we assume there is no ToC in second half of doc
            break
        results = extract_columns(page, x_tolerance, y_tolerance)
        for x, sorted_numbers, is_increasing, row_texts in results:
            # Score based on column length and alignment with page length
            if sorted_numbers and is_increasing:
                page_text = page.get_text().lower()
                length_score = len(sorted_numbers)
                placement_score = 10/(page.number+1)
                page_count_score = 8/(abs(sorted_numbers[-1]-doc.page_count)+1) # how close to the end of doc the potential ToC ends
                begin_score = 8/(abs(sorted_numbers[0])+1) # how close begining page is to begining of doc
                # Check for keywords in the texts
                keyword_score = sum( value for key, value in keywords.items() if key in page_text)

                score = length_score + page_count_score + keyword_score

                if score > best_score:
                    best_score = score
                    best_candidate = (page.number, x, sorted_numbers, row_texts, score)

    return best_candidate

# Usage
#file_path = '../examples/Alphabet_2022.pdf'
dir_path = "../examples/"
directory = os.fsencode(dir_path)
toc_list = []
x_tolerance = 17 
y_tolerance = 5
 # Adjust as needed based on your document's layout
if __name__ == "__main__":
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        full_path = dir_path+filename
        doc = fitz.open(full_path)
        toc = find_best_candidate(doc,x_tolerance,y_tolerance)
        toc_list.append((filename,toc[0]+1,toc[4])) # account for 0-indexing
        print(f"Page number: {toc[0]}")
        print(f"Column at x={toc[1]}:")
        print("Sorted Numbers:", toc[2])
        print("Extracted Text:", toc[3])
        print("Score:",toc[4])
        print("\n")
        doc.close()

#toc_candidate = find_best_candidate(doc, x_tolerance, y_tolerance)
with open('toc_pred.csv','w') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['filename','page_pred','score'])
    csv_out.writerows(toc_list)



if False:
    dir_path = "../examples/"
    directory = os.fsencode(dir_path)
    toc_list = []
    if __name__ == "__main__":
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            full_path = dir_path+filename
            doc = fitz.open(full_path)
            n_tables = 0
            i = 0
            toc = doc.get_toc()
            toc_list.append((filename,toc))

    cnt_empty = 0
    for filename, toc in toc_list:
        print(f"{filename} : {len(toc)}")
        if len(toc) == 0:
            cnt_empty = cnt_empty + 1

    print(f"cnt_empty is {cnt_empty}")




