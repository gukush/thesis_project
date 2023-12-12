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
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score,  calinski_harabasz_score, davies_bouldin_score
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import warnings
import re
#import numpy as np

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

ENLARGE_X = 100
ENLARGE_Y = 100

ENLARGE_CELL_X = 7
ENLARGE_CELL_Y = 7

keywords = ['consolidated','financial','statement','cash','flow']

def plot_results(pil_img, scores, labels, boxes,table_desc, filename, model,structure=True):
    plt.clf()
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        if structure and model.config.id2label[label] in ["table column","table row"]:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
            text = f'{model.config.id2label[label]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
        elif not structure and model.config.id2label[label] == "table":
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
            text = f'{model.config.id2label[label]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    rand_bytes = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    plt.savefig(f'../images/{filename}_{table_desc}.png')



result_list = []
dir_path = "../examples/"
#mat = fitz.Matrix(300/72 , 300/72)
mat = fitz.Matrix(10,10)
#feature_extractor = DetrFeatureExtractor()
directory = os.fsencode(dir_path)

def initializeTable():
    model_structure = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    model_detection = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    return model_structure, model_detection, image_processor
#user_input = input("Type table to search for: ")

def initializeTableSlim():
    model_detection = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    return model_detection, image_processor


def find_optimal_clusters(data, max_k=30):
    scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        score = silhouette_score(data, kmeans.labels_) # todo, maybe compare scores?
        #score = calinski_harabasz_score(data, kmeans.labels_)
        #score = davies_bouldin_score(data, kmeans.labels_)
        scores.append(score)
    optimal_k = scores.index(max(scores)) + 2  # Adding 2 because range starts from 2
    return optimal_k

DENSITY_FACTOR = 2 # bigger if we want to handle sparser tables

def ClusterInto(spans,max_k_y=20,max_k_x=30):
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*value of `n_init` will change from 10 to.*")
    #warnings.filterwarnings("ignore",category=ConvergenceWarning,message=".*found smaller than n_clusters.*")
    # Extracting center points of each bbox
    centers = np.array([(0.5 * d['bbox'][0] + 0.5 * d['bbox'][2], 0.5 * (d['bbox'][1] + d['bbox'][3])) for d in spans])
    max_k_x = len(np.unique(centers[:,0]))
    max_k_y = len(np.unique(centers[:,1]))
    print(f"max_k_x: {max_k_x}, max_k_y: {max_k_y}")
    #if max_k_y < 0:
    #    max_k_y = int(DENSITY_FACTOR*(len(spans)//optimal_k_x)) # we assume that table is dense
        # enough that # cols would be proportional to ()# observations)/(# cols)
    if max_k_x > 1:
        optimal_k_x = find_optimal_clusters(centers[:, 0].reshape(-1, 1),max_k_x)     
        kmeans_x = KMeans(n_clusters=optimal_k_x, random_state=0,n_init='auto').fit(centers[:, 0].reshape(-1, 1))
        labels_x = kmeans_x.labels_
    else:
        labels_x = np.zeros(len(spans), dtype=int)
        optimal_k_x = 1
    if max_k_y > 1:
        optimal_k_y = find_optimal_clusters(centers[:, 1].reshape(-1, 1),max_k_y)
        kmeans_y = KMeans(n_clusters=optimal_k_y, random_state=0,n_init='auto').fit(centers[:, 1].reshape(-1, 1))
        labels_y = kmeans_y.labels_
    else:
        labels_y = np.zeros(len(spans), dtype=int)
        optimal_k_y = 1
    # Clustering X and Y coordinates separately
    # Creating the AxB grid
    
    print(f"optimal k_y {optimal_k_y}")
    print(f"optimal k_x {optimal_k_x}")
    if optimal_k_x > 1:
        centroids_x = kmeans_x.cluster_centers_
        sorted_indices_x = np.argsort(centroids_x.ravel())
    else:
        sorted_indices_x = np.array([0])
    if optimal_k_x > 1:
        centroids_y = kmeans_y.cluster_centers_
        sorted_indices_y = np.argsort(centroids_y.ravel())
    else:
        sorted_indices_y = np.array([0])
    # Creating a sorted grid
    sorted_matrix = [[[] for _ in range(optimal_k_x)] for _ in range(optimal_k_y)]
    for item, label_x, label_y in zip(spans, labels_x, labels_y):

        #for item in spans:
        #    center = (0.5 * (item['bbox'][0] + item['bbox'][2]), 0.5 * (item['bbox'][1] + item['bbox'][3]))
        #cluster_x = kmeans_x.predict([[center[0]]])[0]
        #cluster_y = kmeans_y.predict([[center[1]]])[0]
        sorted_x = np.where(sorted_indices_x == label_x)[0][0]
        sorted_y = np.where(sorted_indices_y == label_y)[0][0]
        sorted_matrix[sorted_y][sorted_x].append(item['text'])
    joined_matrix = [[' '.join(sorted_matrix[i][j]).strip() for j in range(len(sorted_matrix[i]))] for i in range(len(sorted_matrix))]
    #print(joined_matrix)
    return joined_matrix

MIN_Y_SIZE = 5
MIN_X_SIZE = 7
def clean_matrix(matrix):
    cleaned_matrix = []
    number_separator_regex = re.compile(r'(\d+[\.,]?\d*)\s+(\(?\d+[\.,]?\d*\)?)')
    for row in matrix:
        # Filter out empty strings and append the cleaned row
        cleaned_row = list(filter(None, row))
        new_row = []
        for cell in cleaned_row:
            if cell == '$':
                continue
            # Find matches and split them into different rows
            matches = number_separator_regex.match(cell)
            if matches:
                new_row.extend(matches.groups())
            else:
                new_row.append(cell)
        
        cleaned_matrix.append(new_row)
    return cleaned_matrix

def GetMatrixFromClustering(page,mat,box):
    new_box = box.tolist()
    new_area = fitz.Rect(new_box[0] - ENLARGE_X, new_box[1]-ENLARGE_Y, new_box[2]+ENLARGE_X, new_box[3]+ENLARGE_Y)        
    original_box = new_area*~mat
    big_dict = page.get_text("dict",clip = original_box)
    #print(big_dict.get('blocks'))
    spans = [span for block in big_dict.get('blocks',[]) # dollars in random places break things
        for line in block.get('lines',[])
        for span in line.get('spans',[])]
    spans = [span for span in spans if span['text'] != '$']
    #print(spans)
    #spans = [ span for block in data.get('blocks', [])
    #    for line in block.get('lines', [])
    #    for span in line.get('spans', [])
    #    ]
    #max_k_y = 30
    max_k_x = 30 # we assume that tables are vertical and that maximum amount of columns is 30
    matrix = ClusterInto(spans)
    matrix = clean_matrix(matrix)
    print(matrix)
    return matrix


def GetMatrixFromStructure(page,mat,rows,cols,origin_x, origin_y):
    # sort rows according to y0 and cols according to x0
    rows.sort(key = lambda x: x[1])
    cols.sort(key = lambda x: x[0])
    matrix = [[None for _ in cols] for _ in rows]
    for i, row in enumerate(rows):
        #print("___________________NEW_ROW________________________\n")
        for j, col in enumerate(cols):
            box = fitz.Rect(col[0]+origin_x - ENLARGE_CELL_X,row[1]+origin_y-ENLARGE_CELL_Y,col[2]+origin_x+ENLARGE_CELL_X,row[3]+origin_y+ENLARGE_CELL_Y)
            original_box = box*~mat
            #print(f"________________ORIGINAL_BOX:{original_box}______\n")
            #print("___________________NEW_COL________________________\n")
            #dicts = page.get_text('dict',clip=original_box)
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
def ExtractTable(img, box, dx, dy, model, image_processor):
    box = box.tolist()
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

SPLIT = True


def ClusterTableStepByStep(page,mat,model_detection,image_processor,plotting=False, basename="plot",cnt=0):
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
    results = SearchForTable(img, image_processor, model_detection)
    tables_matrix_form = []
    if plotting:
        plot_results(img,results['scores'],results['labels'],results['boxes'],f"page_{page.number}",basename,model_detection,False)
    for score, label, box in zip(results["scores"],results["labels"],results["boxes"]):
        if model_detection.config.id2label[label.item()] == "table":
            print("Table detected! Trying to cluster elements")
            cnt = cnt + 1
            our_matrix = GetMatrixFromClustering(page,mat,box)
            tables_matrix_form.append(our_matrix)
    return tables_matrix_form, cnt

# gets fitz page object and searches for tables in it, dumping them to file-like object if they get found
# returns list of tables which are represented like matrices (lists of lists)
def TableStepByStep(page,mat,model_detection, model_structure, image_processor, plotting = False, basename="plot",cnt=0):
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
    results = SearchForTable(img, image_processor, model_detection)
    tables_matrix_form = []
    #if plotting:
    plot_results(img,results['scores'],results['labels'],results['boxes'],f"page_{page.number}",basename,model_detection,False)
    for score, label, box in zip(results["scores"],results["labels"],results["boxes"]):
        if model_detection.config.id2label[label.item()] == "table":
            print("Table detected!")
            cnt = cnt + 1
            new_results = ExtractTable(img,box,ENLARGE_X,ENLARGE_Y,model_structure,image_processor) #image_processor.post_process_object_detection(new_outputs, threshold=0.9,target_sizes=new_target_sizes)[0]
            if plotting:
                box = box.tolist()
                new_area = (box[0] - ENLARGE_X, box[1]-ENLARGE_Y, box[2]+ENLARGE_X, box[3]+ENLARGE_Y)
                # we need to remember origin coordinates of new area to later select right text from page 
                new_img = img.crop(new_area)
                plot_results(new_img,new_results['scores'],new_results['labels'],new_results['boxes'],cnt,basename,model_structure)
            
            rows = []
            cols = []
            for new_score, new_label, new_box in zip(new_results["scores"],new_results["labels"],new_results['boxes']):
                if model_structure.config.id2label[new_label.item()] == "table row":
                    rows.append(new_box)
                if model_structure.config.id2label[new_label.item()] == "table column":
                    cols.append(new_box)
            our_matrix = GetMatrixFromStructure(page,mat,rows,cols,box[0]-ENLARGE_X,box[1]-ENLARGE_Y)
            tables_matrix_form.append(our_matrix)
    return tables_matrix_form, cnt

def TableExtractionFromStream(stream, keywords, model_detection, model_structure, image_processor, pix_mat = mat, plotting = False, num_start = None, num_end = None):
    doc = fitz.Document(stream=stream)
    if num_start is None:
        num_start = 0
    if num_end is None:
        num_end = doc.page_count
    csv_strings = []
    cnt_table = 0
    for i in range(num_start,num_end):
        page = doc[i]
        page_text = page.get_text("text")
        extract = any(keyword.lower() in page_text.lower() for keyword in keywords)
        tables = []
        if extract:
            tables, cnt_table = ClusterTableStepByStep(page,pix_mat,model_detection,image_processor,plotting,'false',cnt_table)#TableStepByStep(page,pix_mat,model_detection,model_structure,image_processor,plotting)
        for table in tables:
            csv_string = io.StringIO()
            SimpleDumpCSV(csv_string,table)
            csv_strings.append((i,csv_string.getvalue())) # tuple of Page numbers as well as csv contents
            csv_string.close()
    return csv_strings
            
    
# horizontal option is boolean and if true then the images are split horizontally
# if false then they are split vertically
def SplitImageInHalf(img,horizontal = True):
    width, height = img.size
    if horizontal:
        lbox = (0,0, width, height // 2)
        rbox = (0, height // 2, width, height)
    else:
        lbox = (0,0,width // 2, height)
        rbox = (width // 2, 0, width, height)

    img_l = img.crop(lbox)
    img_r = img.crop(rbox)
    return img_l, img_r

if __name__ == "__main__":
    df = pd.read_csv("../table_test_pages.csv")
    model_structure, model_detection, image_processor = initializeTable()
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        full_path = dir_path+filename
        basename = os.path.splitext(filename)[0]
        print(f"{basename}")
        start_time = time.time()
        doc = fitz.open(full_path)
        n_tables = 0
        i = 0
        #doc = []
        entry  = df[df['Filename']==filename].iloc[0]
        print(f"Entry:{entry}")
        cnt_table = 0
        for page in doc:
            i = i + 1 
            if os.path.isfile(f"../images/{basename}_page_{page.number}.png"):
                print(f"Following page was already extracted: {page.number} in {basename}")
            elif i in range(entry['Start'],entry['Stop']):
                print(f"{basename}, page {i}")
                page_text = page.get_text("text")
                #extract = any(keyword.lower() in page_text.lower() for keyword in keywords)
                extract = True 
                model_structure, model_detection, image_processor = initializeTable()
                if extract:
                    our_matrices, cnt_table = ClusterTableStepByStep(page,mat,model_detection,image_processor,True, basename,cnt_table)
    #TableStepByStep(page,mat, model_detection, model_structure, image_processor,True,basename,cnt_table)
                    for matrix in our_matrices:
                        n_tables = n_tables + 1
                        with open(f"matrix_{basename}_{n_tables}.csv", "w+") as my_csv:
                            SimpleDumpCSV(my_csv,matrix)
            time_elapsed = time.time() - start_time
            result_list.append((filename,n_tables,time_elapsed))
    for entry in result_list:
        print(entry)


