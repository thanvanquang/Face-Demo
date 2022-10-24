import os
import cv2
import json
import numpy as np
from onnx_utils.features_implement import FacesFeatures

# Load model
x = FacesFeatures(recognize_model='backbone.onnx', detection_model='model.onnx', use_gpu=True)
assets_path = os.path.join('./assets')
x.load(assets_path)

def registration(path_data):
    txt_file = "DB/Haugiang_backbone_mask.txt"
    f = open(txt_file, 'r')
    data_set = json.loads(f.read())

    files = os.listdir(path_data)
    files = sorted(files)
    number = 0
    for file in files:
        file_path = os.path.join(path_data, file)
        person_name = file
        if os.path.isfile(file_path):
            continue
        images = os.listdir(file_path)
        features_arr = []
        for img in images:
            # if img.find("cloth") > -1 or img.find("surgical") > -1:
            #     continue
            if img.find("mask") > -1:           # find and remove mask face
                continue
            img = os.path.join(file_path, img)
            try:
                image = cv2.imread(img)
            except:
                continue
            feature = x.get_feature(image, one_face=True)
            if len(feature) > 0:
                features_arr.append(feature)

        print("Person name %s have %.f images:" % (person_name, len(features_arr)))
        if len(features_arr) > 0:
            features_arr = np.array(features_arr).tolist()
            data_set[person_name] = features_arr
            f = open(txt_file, 'w')
            f.write(json.dumps(data_set))
            number += 1

    print("Total people in DB is %.f" % number)


path_data = "/media/ubuntu/DATA/Trienkhai/HG-register/Tonghop"
registration(path_data)