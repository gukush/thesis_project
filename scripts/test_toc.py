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


