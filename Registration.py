import os
import cv2
import json
import numpy as np
from retina.retinaface_cov import RetinaFaceCoV
from api_usage.Load_model import Loadmodel
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

# TODO: DETECT
# define detector
def detect(img,detector):
    thresh = 0.95
    scales = [640, 1080]
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [im_scale]
    flip = False
    faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
    return faces, landmarks

def getPos(points):
    if abs(points[0] - points[2]) / abs(points[1] - points[2]) > 2:
        return "Right"
    elif abs(points[1] - points[2]) / abs(points[0] - points[2]) > 2:
        return "Left"
    return "Center"

def registration(path_data):
    gpuid = 0
    detector = RetinaFaceCoV('retina/model/cov2/mnet_cov2', 0, gpuid, 'net3l')
    faceRecModelHandler = Loadmodel()
    face_cropper = FaceRecImageCropper()

    txt_file = "./Dongnai.txt"

    f = open(txt_file, 'r')
    data_set = json.loads(f.read())

    files = os.listdir(path_data)
    files = sorted(files)
    number = 0
    for file in files:
        file_path = os.path.join(path_data, file)
        person_name = file
        images = os.listdir(file_path)
        features_arr = []
        for img in images:
            img = os.path.join(file_path, img)
            try:
                image = cv2.imread(img)
            except:
                continue
            bounding_boxes, landmarks_fivepoints = detect(image, detector)

            if bounding_boxes is not None and len(bounding_boxes) > 0:
                area = (bounding_boxes[:, 2] - bounding_boxes[:, 0]) * (bounding_boxes[:, 3] - bounding_boxes[:, 1])
                bindex = np.argmax(area)
                landmarks = landmarks_fivepoints[bindex]
                cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
                feature = faceRecModelHandler.inference_on_image(cropped_image)
                features_arr.append(feature)

        print("Person name %s have %.f images:" % (person_name, len(features_arr)))
        if len(features_arr) > 0:
            features_arr = np.array(features_arr).tolist()
            data_set[person_name] = features_arr
            f = open(txt_file, 'w')
            f.write(json.dumps(data_set))
            number += 1

    print("Total people in DB is %.f" % number)


path_data = "/media/ubuntu/DATA/Dongnai/AI CBGVNV-20220328T045145Z-001/AI CBGVNV/Tonghop"
registration(path_data)