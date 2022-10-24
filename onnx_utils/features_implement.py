import cv2
import numpy as np
import os
from numpy.linalg import norm as l2norm
from skimage import transform as trans
from .scrfd_processing import ScrfdProcessing
import os.path as osp
import onnxruntime
import onnx
from onnx import numpy_helper


arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]], dtype=np.float32)

def estimate_norm(lmk):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, arcface_src)
    M = tform.params[0:2, :]
    return M

def norm_crop(img, landmark, image_size=112):
    Matrix = estimate_norm(landmark)
    warped = cv2.warpAffine(img, Matrix, (image_size, image_size), borderValue=0.0)
    return warped

class FacesFeatures:
    def __init__(self, recognize_model='iR100.onnx', detection_model='model.onnx', conf_detect_thresh=0.4, use_gpu=False):
        self.recognize_model = recognize_model
        self.detection_model = detection_model
        self.conf_detect_thresh = conf_detect_thresh
        self.det_size = 224
        self.use_gpu = use_gpu
        print("use_gpu: {}".format(use_gpu))

    def load(self, rdir):
        det_model = os.path.join(rdir, 'det', self.detection_model)
        self.detector = ScrfdProcessing(path_model=det_model, input_size=(640, 640), conf_thresh=self.conf_detect_thresh)
        print('use onnx-model:', det_model)
        self.model_file = os.path.join(rdir, 'face_reg', self.recognize_model)
        print('use onnx-model:', self.model_file)

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if self.use_gpu:
            session = onnxruntime.InferenceSession(self.model_file, providers=providers)
        else:
            sessionOptions = onnxruntime.SessionOptions()
            sessionOptions.intra_op_num_threads = 1
            sessionOptions.inter_op_num_threads = 1
            session = onnxruntime.InferenceSession(self.model_file, sess_options=sessionOptions, providers=providers)
            print("face reg intra_op_num_threads {} inter_op_num_threads {}".format(sessionOptions.intra_op_num_threads,sessionOptions.inter_op_num_threads))


        input_cfg = session.get_inputs()[0]
        input_shape = input_cfg.shape
        print('input-shape:', input_shape)
        if len(input_shape) != 4:
            return "length of input_shape should be 4"
        if not isinstance(input_shape[0], str):
            print('reset input-shape[0] to None')
            model = onnx.load(self.model_file)
            model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
            new_model_file = osp.join(self.model_path, 'zzzzrefined.onnx')
            onnx.save(model, new_model_file)
            self.model_file = new_model_file
            print('use new onnx-model:', self.model_file)
            try:
                session = onnxruntime.InferenceSession(self.model_file, None)
            except:
                return "load onnx failed"

            input_cfg = session.get_inputs()[0]
            input_shape = input_cfg.shape
            print('new-input-shape:', input_shape)

        self.image_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        if len(output_names) != 1:
            return "number of output nodes should be 1"
        self.session = session
        self.input_name = input_name
        self.output_names = output_names
        model = onnx.load(self.model_file)
        graph = model.graph
        if len(graph.node) < 8:
            return "too small onnx graph"
        self.crop = None
        input_mean = None
        input_std = None
        if input_mean is not None or input_std is not None:
            if input_mean is None or input_std is None:
                return "please set input_mean and input_std simultaneously"
        else:
            find_sub = False
            find_mul = False
            for nid, node in enumerate(graph.node[:8]):
                print(nid, node.name)
                if "sub" in node.name.lower() or "minus" in node.name.lower():
                    find_sub = True
                if "mul" in node.name.lower() or "div" in node.name.lower():
                    find_mul = True

            print("find_sub {} find_mul {}".format(find_sub, find_mul))
            if find_sub and find_mul:
                # mxnet arcface model
                input_mean = 0.0
                input_std = 1.0
            else:
                input_mean = 127.5
                input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        for initn in graph.initializer:
            weight_array = numpy_helper.to_array(initn)
            dt = weight_array.dtype
            if dt.itemsize < 4:
                return 'invalid weight type - (%s:%s)' % (initn.name, dt.name)

    def detect(self, img):
        bboxes, landmarks = self.detector.run(img)
        return bboxes, landmarks

    def get_feature(self, img, one_face=True):
        bboxes, det_landmarks = self.detect(img)
        if bboxes.shape[0] == 0:
            if one_face:
                return []
            else:
                return [], []
        det = bboxes
        if one_face:
            area = (det[:, 2]-det[:, 0])*(det[:, 3]-det[:, 1])
            box_cw = (det[:, 2]+det[:, 0]) / 2
            box_ch = (det[:, 3]+det[:, 1]) / 2
            dist_cw = box_cw - img.shape[1]/2
            dist_ch = box_ch - img.shape[0]/2
            score = area - (dist_cw**2 + dist_ch**2)*2.0
            bindex = np.argmax(score)
            det_landmark = det_landmarks[bindex]
            aimg = norm_crop(img, det_landmark)
            input_size = self.image_size
            if not isinstance(aimg, list):
                aimg = [aimg]
            blob = cv2.dnn.blobFromImages(aimg, 1.0, input_size, (0, 0, 0), swapRB=True)
            net_out = self.session.run(None, {self.input_name: blob})[0]
            feat = np.mean(net_out, axis=0)
            feat /= l2norm(feat)
            return feat
        else:
            feat_faces = []
            for i in range(bboxes.shape[0]):
                det_landmark = det_landmarks[i]
                aimg = norm_crop(img, det_landmark)

                input_size = self.image_size
                if not isinstance(aimg, list):
                    aimg = [aimg]
                blob = cv2.dnn.blobFromImages(aimg, 1.0, input_size, (0, 0, 0), swapRB=True)
                net_out = self.session.run(None, {self.input_name: blob})[0]
                feat = np.mean(net_out, axis=0)
                feat /= l2norm(feat)
                feat_faces.append(feat)
            return feat_faces, det

    def get_sim(self, feat1, feat2):
        return np.dot(feat1, feat2)
