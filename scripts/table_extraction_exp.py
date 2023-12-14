import table_extraction as tab 
import fitz
import os

file = 'AMD_10K_2021.pdf'
dir = '../examples/'

# wczesniej Munich Re strona 136

doc = fitz.open(os.path.join(dir,file))
num_page = 54
page = doc.load_page(num_page)


model_structure, model_detection, image_processor = tab.initializeTable()


mat = fitz.Matrix(10,10)
if True:
    matrix_results,cnt = tab.ClusterTableStepByStep(page,mat,model_detection, 
                                                    image_processor ,plotting=True)
    n_tables = 0
    for matrix in matrix_results:
        n_tables = n_tables + 1
        with open(f"matrix_{file.split('.')[0]}_{n_tables}_cluster.csv", "w+") as my_csv:
            tab.SimpleDumpCSV(my_csv,matrix)

if True:
    matrix_results,cnt = tab.TableStepByStep(
        page,mat, model_detection, model_structure, image_processor, plotting=True)
    
    n_tables = 0
    for matrix in matrix_results:
            n_tables = n_tables + 1
            with open(f"matrix_{file.split('.')[0]}_{n_tables}_transformer.csv", "w+") as my_csv:
                tab.SimpleDumpCSV(my_csv,matrix)

