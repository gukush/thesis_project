import table_extraction as tab 
import fitz
import os

file = 'apple_2020.pdf'
dir = '../examples/'

doc = fitz.open(os.path.join(dir,file))
num_page = 33
page = doc.load_page(num_page)


model_detection, image_processor = tab.initializeTableSlim()


mat = fitz.Matrix(10,10)
matrix_results,cnt = tab.ClusterTableStepByStep(page,mat,model_detection, image_processor)
n_tables = 0
for matrix in matrix_results:
    n_tables = n_tables + 1
    with open(f"matrix_{file.split('.')[0]}_{n_tables}_cluster.csv", "w+") as my_csv:
        tab.SimpleDumpCSV(my_csv,matrix)
