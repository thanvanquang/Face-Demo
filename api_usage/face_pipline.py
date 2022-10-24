import sys
sys.path.append('.')
import os
import yaml
import cv2
import numpy as np
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

with open('../config/model_conf.yaml') as f:
    model_conf = yaml.load(f)

if __name__ == '__main__':
    # common setting for all models, need not modify.
    model_path = '../models'

    # face detection model setting.
    scene = 'mask'
    model_category = 'face_detection'
    model_name = model_conf[scene][model_category]
    print('Start to load the face detection model...')
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        print('Falied to load face detection Model.')
        sys.exit(-1)
    else:
       print('Success!')

    # face landmark model setting.
    model_category = 'face_alignment'
    model_name = model_conf[scene][model_category]
    print('Start to load the face landmark model...')
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
        model, cfg = faceAlignModelLoader.load_model()
        faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        print('Failed to load face landmark model.')
        sys.exit(-1)
    else:
        print('Success!')

    # face recognition model setting.
    model_category = 'face_recognition'
    model_name = model_conf[scene][model_category]
    print('Start to load the face recognition model...')
    try:
        faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
        model, cfg = faceRecModelLoader.load_model()
        faceRecModelHandler = FaceRecModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        print('Failed to load face recognition model.')
        sys.exit(-1)
    else:
        print('Success!')

    # read image and get face features.
    # compare from 2 images

    face_cropper = FaceRecImageCropper()
    images_path = "/home/ubuntu/DATA/CNL_test/CNL_test/2faces"
    images = os.listdir(images_path)
    feature_list = []
    try:
        for image_name in images:
            image_path = os.path.join(images_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            dets = faceDetModelHandler.inference_on_image(image)
            face_nums = dets.shape[0]
            if face_nums != 1:
                print('Input image should contain 1 face!')
            for i in range(face_nums):
                landmarks = faceAlignModelHandler.inference_on_image(image, dets[i])
                landmarks_list = []
                for (x, y) in landmarks.astype(np.int32):
                    landmarks_list.extend((x, y))
                cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)
                feature = faceRecModelHandler.inference_on_image(cropped_image)
                feature_list.append(feature)
        score = np.dot(feature_list[0], feature_list[1])
        distance = np.linalg.norm(feature_list[0]-feature_list[1])
        print('Total faces: %.f' % len(feature_list))
        print('The similarity score of two faces: %f' % score)
        print('The distance of two faces: %f' % distance)
    except Exception as e:
        print('Pipeline failed!')
        sys.exit(-1)
    else:
        print('Success!')
