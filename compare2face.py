import os
import cv2
import json
import numpy as np
from onnx_utils.features_implement import FacesFeatures


# Load model
x = FacesFeatures(recognize_model='backbone.onnx', detection_model='model.onnx', use_gpu=True)
assets_path = os.path.join('./assets')
x.load(assets_path)

def euclid_distance(emb1, emb2):
    return np.linalg.norm(emb1-emb2, keepdims=False)

path_1 = ""
path_2 = ""

image_1 = cv2.imread(path_1)
image_2 = cv2.imread(path_2)

feature_1 = x.get_feature(image_1, one_face=True)
feature_2 = x.get_feature(image_2, one_face=True)

cosin_distance = euclid_distance(feature_1, feature_2)

print(cosin_distance)