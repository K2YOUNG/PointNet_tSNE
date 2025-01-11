import os, json
import numpy as np
from plyfile import PlyData, PlyElement
import pandas as pd

def sort_by_object(scan_dir, scene_name, cat_pth="category_mapping.tsv"):
    ply_path = os.path.join(scan_dir, f"{scene_name}.ply")

    plydata = PlyData.read(ply_path)
    category_tsv = pd.read_csv(cat_pth, sep="\t", encoding = "utf-8")

    obj_labels = np.zeros(len(plydata.elements[0]))

    for face in plydata.elements[1]:
        obj_raw_cat = face[-1]
        if obj_raw_cat == -1:
            obj_raw_cat = 7
        points_indices = face[0]

        for idx in points_indices:
            obj_labels[int(idx)] = category_tsv.loc[obj_raw_cat, 'mpcat40index']
        
    np.save("2t7WUuJeko7_points_labels.npy", obj_labels)

    
if __name__ == "__main__":
    sort_by_object('.', "2t7WUuJeko7")