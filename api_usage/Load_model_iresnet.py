import sys
sys.path.append('.')
import yaml
import torch
from models.network_def.iresnet import iresnet50
from core.model_handler.face_recognition.FaceRecModelHandler_iresnet import FaceRecModelHandler

def Loadmodel():
    cfg = "quang"
    model_path = "models/face_recognition_iresnet/backbone.pth"
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    print('Start to load the face recognition model...')
    try:
        weight = torch.load(model_path, map_location=device)
        model = iresnet50(False).cuda()
        model.load_state_dict(weight)
        model.to(device)
        model.eval()
        faceRecModelHandler = FaceRecModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        print('Failed to load face recognition model.')
        sys.exit(-1)
    else:
        print('Success!')
    return faceRecModelHandler