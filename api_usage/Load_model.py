import sys
sys.path.append('.')
import yaml
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

def Loadmodel():
    with open('config/model_conf.yaml') as f:
        model_conf = yaml.load(f)
    model_path = 'models'

    # face detection model setting.
    scene = 'mask'
    # model_category = 'face_detection'
    # model_name = model_conf[scene][model_category]
    # print('Start to load the face detection model...')
    # try:
    #     faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
    #     model, cfg = faceDetModelLoader.load_model()
    #     faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)
    # except Exception as e:
    #     print('Falied to load face detection Model.')
    #     sys.exit(-1)
    # else:
    #     print('Success!')

    # face landmark model setting.
    # model_category = 'face_alignment'
    # model_name = model_conf[scene][model_category]
    # print('Start to load the face landmark model...')
    # try:
    #     faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
    #     model, cfg = faceAlignModelLoader.load_model()
    #     faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)
    # except Exception as e:
    #     print('Failed to load face landmark model.')
    #     sys.exit(-1)
    # else:
    #     print('Success!')

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

    return faceRecModelHandler