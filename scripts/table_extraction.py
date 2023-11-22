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
import io
#import numpy as np

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

ENLARGE_X = 100
ENLARGE_Y = 100

keywords = ['consolidated statement']

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
#user_input = input("Type table to search for: ")


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


def SearchForTable(img, image_processor, model):
    inputs = image_processor(images=img,return_tensors="pt")
    with torch.no_grad():
        #should be model for detection
        outputs = model(**inputs)
    target_sizes = torch.tensor([img.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold = 0.9, target_sizes=target_sizes)[0]
    return results
# dx and dy in this function are variables responsible for how much area outside of the box should also be taken from the image
def ExtractTable(img, box, dx, dy, model):
    new_area = (box[0] - dx, box[1]-dy, box[2]+dx, box[3]+dy)
     # we need to remember origin coordinates of new area to later select right text from page 
    new_img = img.crop(new_area)
    inputs = image_processor(new_img, return_tensors="pt")
    with torch.no_grad():
    	#should be model for structure analysis
    	outputs = model(**inputs)
    target_sizes = torch.tensor([new_img.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9,target_sizes=target_sizes)[0]
    #plot_results(new_img,new_results['scores'],new_results['labels'],new_results['boxes'],n_tables)
    return results

def SimpleDumpCSV(file_like, matrix):
    csvWriter = csv.writer(file_like,delimiter=',')
    csvWriter.writerows(matrix)

# gets fitz page object and searches for tables in it, dumping them to file-like object if they get found
# returns list of tables which are represented like matrices (lists of lists)
def TableStepByStep(page,mat,model_detection, model_structure, plotting = False):
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
    results = SearchForTable(img, image_processor, model_detection)
    tables_matrix_form = []
    for score, label, box in zip(results["scores"],results["labels"],results["boxes"]):
        if model_detection.config.id2label[label.item()] == "table":
            new_results = ExtractTable(img,box,ENLARGE_X,ENLARGE_Y,model_structure) #image_processor.post_process_object_detection(new_outputs, threshold=0.9,target_sizes=new_target_sizes)[0]
            if plotting:
                plot_results(new_img,new_results['scores'],new_results['labels'],new_results['boxes'],n_tables)
            rows = []
            cols = []
            for new_score, new_label, new_box in zip(new_results["scores"],new_results["labels"],new_results['boxes']):
                if model_structure.config.id2label[new_label.item()] == "table row":
                    rows.append(new_box)
                if model_structure.config.id2label[new_label.item()] == "table column":
                    cols.append(new_box)
                our_matrix = GetMatrixFromStructure(page,mat,rows,cols,box[0]-ENLARGE_X,box[1]-ENLARGE_Y)
                tables_matrix_form.append(our_matrix)
    return tables_matrx_form

def TableExtractionFromStream(stream, keywords, pix_mat=mat, model_detection=model_detection, model_structure=model_structure, plotting = False, num_start = None, num_end = None):
    doc = fitz.Document(stream=stream)
    if num_start is None:
        num_start = 0
    if num_end is None:
        num_end = doc.page_count 
    for i in range(num_start,num_end):
        page = doc[i]
        page_text = page.get_text("text")
        extract = any(keyword in page_text for keyword in keywords)
        print(f"extract {extract}")
        tables = []
        if extract:
            tables = TableStepByStep(page,pix_mat,model_detection,model_structure,plotting)
        csv_strings = []
        for table in tables:
            csv_string = io.StringIo()
            SimpleDumpCSV(csv_string,table)
            csv_strings.append(csv_string)
    return csv_strings
            
    

if __name__ == "__main__":
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        full_path = dir_path+filename
        start_time = time.time()
        doc = fitz.open(full_path)
        n_tables = 0
        for page in doc:
            page_text = page.get_text("text")
            if user_input.lower() in page_text.lower():
                our_matrices = TableStepByStep(page,mat, model_detection, modle_structure, True)
                for matrix in our_matrices:
                    n_tables = n_tables + 1
                    with open(f"test_matrix{n_tables}.csv", "w+") as my_csv:
                        SimpleDumpCSV(my_csv,our_matrix)
        time_elapsed = time.time() - start_time
        result_list.append((filename,n_tables,time_elapsed))
    for entry in result_list:
        print(entry)


