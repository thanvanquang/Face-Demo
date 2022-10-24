import cv2
import numpy as np
import os
from skimage import transform as trans
from .scrfd_processing import ScrfdProcessing
import onnxruntime
import time


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M

def softmax(x):
    x = np.exp(x) / sum(np.exp(x))
    return x



class FacesFeatures:
    def __init__(self, detection_model='model.onnx', agegender_model='model.onnx', emotion_model='model.onnx',
                 conf_detect_thresh=0.4, num_emotions=7, use_gpu=False):
        self.detection_model = detection_model
        self.conf_detect_thresh = conf_detect_thresh
        self.agegender_model = agegender_model
        self.emotion_model = emotion_model
        self.use_gpu = use_gpu
        print("use_gpu: {}".format(use_gpu))

        if num_emotions == 7:
            self.emotion_idx_to_class = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness',
                            6: 'Surprise'}
        else:
            self.emotion_idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral',
                            6: 'Sadness',
                            7: 'Surprise'}

        self.gender_idx_to_class = {0: 'Female', 1: 'Male'}


    def load(self, rdir):
        # Load detection model
        det_model = os.path.join(rdir, 'det', self.detection_model)
        self.detector = ScrfdProcessing(path_model=det_model, input_size=(640, 640), conf_thresh=self.conf_detect_thresh)
        print('use det onnx-model:', det_model)

        # Load emotion model
        self.emotion_model_file = os.path.join(rdir, 'emotion', self.emotion_model)
        print('use emotion onnx-model:', self.emotion_model_file)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if self.use_gpu:
            emotion_session = onnxruntime.InferenceSession(self.emotion_model_file, providers=providers)
        else:
            sessionOptions = onnxruntime.SessionOptions()
            sessionOptions.intra_op_num_threads = 1
            sessionOptions.inter_op_num_threads = 1
            emotion_session = onnxruntime.InferenceSession(self.emotion_model_file, sess_options=sessionOptions, providers=providers)

        emotion_input_cfg = emotion_session.get_inputs()[0]
        emotion_input_shape = emotion_input_cfg.shape
        print('input-shape:', emotion_input_shape)

        self.emotion_image_size = tuple(emotion_input_shape[2:4][::-1])
        emotion_input_name = emotion_input_cfg.name
        emotion_outputs = emotion_session.get_outputs()
        emotion_output_names = []
        for o in emotion_outputs:
            emotion_output_names.append(o.name)
        if len(emotion_output_names) != 1:
            return "number of output nodes should be 1"
        self.emotion_session = emotion_session
        self.emotion_input_name = emotion_input_name
        self.emotion_output_names = emotion_output_names
        self.emotion_input_mean = [0.485, 0.456, 0.406]
        self.emotion_input_std = [0.229, 0.224, 0.225]

        # Load Agegender model
        self.agegender_model_file = os.path.join(rdir, 'agegender', self.agegender_model)
        print('use agegender onnx-model:', self.agegender_model_file)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if self.use_gpu:
            agegender_session = onnxruntime.InferenceSession(self.agegender_model_file, providers=providers)
        else:
            sessionOptions = onnxruntime.SessionOptions()
            sessionOptions.intra_op_num_threads = 1
            sessionOptions.inter_op_num_threads = 1
            agegender_session = onnxruntime.InferenceSession(self.agegender_model_file, sess_options=sessionOptions,
                                                             providers=providers)

        agegender_input_cfg = agegender_session.get_inputs()[0]
        agegender_input_shape = agegender_input_cfg.shape
        print('input-shape:', agegender_input_shape)

        self.agegender_image_size = tuple(agegender_input_shape[2:4][::-1])
        agegender_input_name = agegender_input_cfg.name
        agegender_outputs = agegender_session.get_outputs()
        agegender_output_names = []
        for o in agegender_outputs:
            agegender_output_names.append(o.name)
        if len(agegender_output_names) != 1:
            return "number of output nodes should be 1"
        self.agegender_session = agegender_session
        self.agegender_input_name = agegender_input_name
        self.agegender_output_names = agegender_output_names
        self.agegender_input_mean = 0.0
        self.agegender_input_std = 1.0

    def detect(self, img):
        t0 = time.time()
        bboxes, landmarks = self.detector.run(img)
        print("time run detection model:", time.time() - t0)
        return bboxes, landmarks

    def preprocess_emotion(self, box, img):
        x1, y1, x2, y2 = box[0:4]
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, img.shape[1]), min(y2, img.shape[0])
        aimg = img[y1:y2, x1:x2, :]
        aimg = cv2.resize(aimg, self.emotion_image_size)
        aimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
        aimg = aimg.astype(np.float32) / 255.
        aimg = self.transform(aimg)
        aimg = np.expand_dims(aimg, axis=0)
        return aimg

    def forward_emotion(self, inputs):
        t0_run = time.time()
        net_out = self.emotion_session.run(self.emotion_output_names, {self.emotion_input_name: inputs})[0][0]
        print("time run emotion model:", time.time() - t0_run)
        net_out = softmax(net_out)
        label = self.emotion_idx_to_class[np.argmax(net_out)]
        score = np.max(net_out)
        return label, score

    def preprocess_agegender(self, box, img):
        bbox = box[0:4]
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.agegender_image_size[0] / (max(w, h) * 1.5)
        aimg, M = transform(img, center, self.agegender_image_size[0], _scale, rotate)
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.agegender_input_std, self.agegender_image_size,
                                     (self.agegender_input_mean, self.agegender_input_mean, self.agegender_input_mean),
                                     swapRB=True)
        return blob

    def forward_agegender(self, inputs):
        t0_run = time.time()
        pred = self.agegender_session.run(self.agegender_output_names, {self.agegender_input_name: inputs})[0][0]
        print("time run agegender model:", time.time() - t0_run)
        logit_gender = softmax(pred[:2])
        gender = self.gender_idx_to_class[np.argmax(logit_gender)]
        score = np.max(logit_gender)

        age = int(np.round(pred[2] * 100))
        return gender, score, age


    def run(self, img):
        bboxes, det_landmarks = self.detect(img)
        emotions = []
        agegender = []
        for i in range(bboxes.shape[0]):
            box = bboxes[i].astype(int)
            # run emotion
            input_emotion = self.preprocess_emotion(box, img)
            output_emotion = self.forward_emotion(input_emotion)
            emotions.append(output_emotion)


            # run agegender
            input_agegender = self.preprocess_agegender(box, img)
            output_agegender = self.forward_agegender(input_agegender)
            agegender.append(output_agegender)
        return bboxes, emotions, agegender







    def transform(self, x):
        x = x.transpose(-1, 0, 1)
        x = [(x[i] - self.emotion_input_mean[i]) / self.emotion_input_std[i] for i in range(x.shape[0])]
        x = np.asarray(x)
        return x


