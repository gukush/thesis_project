import fitz
import os
import json


example_file = 'apple_2021.pdf'
example_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../examples/',example_file)



example_page = 36

doc = fitz.open(example_path)
page = doc.load_page(example_page)

area = fitz.Rect(320,320,395,375)
big_dict = page.get_text("dict",clip=area)

json_object = json.dumps(big_dict, indent = 4)
print(json_object)
#print(big_dict['blocks'][2]['lines'])
total_len = 0
total_spans = 0
big_dict = {'blocks':[]}
for block in big_dict['blocks']:
    for line in block['lines']:
        for span in line['spans']:
            total_spans = total_spans + 1
            print("--------------------------")
            print(span['text'])
            print("--------------------------")
            total_len = total_len + len(span['text'])



print(f"Average length of span is {total_len/total_spans} for page {example_page} in {example_file}")

