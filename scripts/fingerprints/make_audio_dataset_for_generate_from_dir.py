import datasets
import sys
import os

file_names = []

for file_name in os.listdir(sys.argv[1]):
    if not file_name.endswith(".wav"):
        print("not wav file", file_name)
    
    file_names.append(file_name)

dataset_items = {
    "file_name": file_names
}

print("file_names", file_names[:10])

datasets.Dataset.from_dict(dataset_items).save_to_disk(sys.argv[2])