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

def extract_columns(doc, page_number):
    page = doc.load_page(page_number)
    blocks = page.get_text("dict")["blocks"]
    
    # Extracting numbers and their positions
    numbers_with_pos = []
    for b in blocks:
        for l in b["lines"]:
            for s in l["spans"]:
                if is_number(s["text"].strip()):
                    numbers_with_pos.append((float(s["text"]), s["bbox"]))

    # Sorting numbers based on vertical position
    numbers_with_pos.sort(key=lambda x: x[1][1])

    # Extracting sorted numbers and checking if they are in increasing order
    sorted_numbers = [num for num, _ in numbers_with_pos]
    is_increasing = all(sorted_numbers[i] <= sorted_numbers[i + 1] for i in range(len(sorted_numbers) - 1))

    # Extracting text in the same horizontal row
    extracted_text = []
    for _, num_pos in numbers_with_pos:
        for b in blocks:
            for l in b["lines"]:
                for s in l["spans"]:
                    if num_pos[1] <= s["bbox"][1] <= num_pos[3] and num_pos[0] != s["bbox"][0]:
                        extracted_text.append(s["text"].strip())

    return sorted_numbers, is_increasing, extracted_text

# Usage
file_path = '../examples/apple_2020.pdf'
doc = fitz.open(file_path)
page_number = 0  # Change as needed

sorted_numbers, is_increasing, extracted_text = extract_columns(doc, page_number)
print("Sorted Numbers:", sorted_numbers)
print("Is Increasing:", is_increasing)
print("Extracted Text:", extracted_text)

doc.close()

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




