import fitz
import random
from transformers import AutoImageProcessor, TableTransformerForObjectDetection, DetrFeatureExtractor
import torch
import os
import time
from PIL import Image
import pandas
import matplotlib.pyplot as plt
import string
import argparse
import csv
#import numpy as np

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

ENLARGE_X = 100
ENLARGE_Y = 100

def plot_results(pil_img, scores, labels, boxes,n_table):
    plt.clf()
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        if model_structure.config.id2label[label] in ["table column","table row"]:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
            text = f'{model_structure.config.id2label[label]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    gowno = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    plt.savefig(f'jajca{gowno}.png')



result_list = []
dir_path = "../examples/"
mat = fitz.Matrix(300/72 , 300/72)
#feature_extractor = DetrFeatureExtractor()
directory = os.fsencode(dir_path)
model_structure = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
model_detection = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
user_input = input("Type table to search for: ")


def GetMatrixFromStructure(page,mat,rows,cols,origin_x, origin_y):
    # sort rows according to y0 and cols according to x0
    rows.sort(key = lambda x: x[1])
    cols.sort(key = lambda x: x[0])
    matrix = [[None for _ in cols] for _ in rows]
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            box = fitz.Rect(col[0]+origin_x,row[1]+origin_y,col[2]+origin_x,row[3]+origin_y)
            original_box = box*~mat
            text = page.get_text("text",clip=original_box).strip()

            matrix[i][j] = text
    return matrix




for file in os.listdir(directory):
    filename = os.fsdecode(file)
    full_path = dir_path+filename
    start_time = time.time()
    doc = fitz.open(full_path)
    n_tables = 0
    for page in doc:
        page_text = page.get_text("text")
        if user_input.lower() in page_text.lower():
    	    pix = page.get_pixmap(matrix=mat)
    	    img = Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
    	    #encoding = feature_extractor(img,return_tensors="pt")
    	    inputs = image_processor(images=img, return_tensors="pt")
    	    outputs = model_detection(**inputs)
    	    #print(encoding.keys())
    	    #with torch.no_grad():
    	    #	outputs = model(**encoding)
    	    target_sizes = torch.tensor([img.size[::-1]])
    	    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    	    for score, label in zip(results["scores"],results["labels"]):
    		    if model_detection.config.id2label[label.item()] == "table":
    		        n_tables = n_tables + 1
    		        print(f"table found with confidence: {round(score.item(),3)}")
    		        for box in results['boxes'].tolist():
    			        new_area = (box[0]-ENLARGE_X,box[1]-ENLARGE_Y,box[2]+ENLARGE_X,box[3]+ENLARGE_Y)
    			        # we need to remember origin coordinates of new area to later select right text from page

    			        new_img = img.crop(new_area)
    			        #new_inputs = image_processor(images=new_img,return_tensors="pt")
    			        new_inputs = image_processor(new_img, return_tensors="pt")
    			        with torch.no_grad():
    			            new_outputs = model_structure(**new_inputs)
    			        new_target_sizes = torch.tensor([new_img.size[::-1]])
    			        new_results = image_processor.post_process_object_detection(new_outputs, threshold=0.9,target_sizes=new_target_sizes)[0]
    			        plot_results(new_img,new_results['scores'],new_results['labels'],new_results['boxes'],n_tables)
    			        rows = []
    			        cols = []
    			        for new_score, new_label, new_box in zip(new_results["scores"],new_results["labels"],new_results['boxes']):
    			            if model_structure.config.id2label[new_label.item()] == "table row":
    			                rows.append(new_box)
    			            if model_structure.config.id2label[new_label.item()] == "table column":
    			                cols.append(new_box)
    			        #print(f"cols: {len(cols)} rows: {len(rows)}")
    			        our_matrix = GetMatrixFromStructure(page,mat,rows,cols,box[0]-ENLARGE_X,box[1]-ENLARGE_Y)
    			        with open(f"test_matrix{n_tables}.csv", "w+") as my_csv:
    			            csvWriter = csv.writer(my_csv,delimiter=',')
    			            csvWriter.writerows(our_matrix)

    time_elapsed = time.time() - start_time
    result_list.append((filename,n_tables,time_elapsed))

for entry in result_list:
    print(entry)


