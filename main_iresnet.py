'''
Main program
@Author: Than Van Quang

To execute simply run:
main.py

To input new user:
main.py --mode "input"
'''

import cv2
import argparse
import sys
import os
import json
from time import time
import datetime
import numpy as np
from retina.retinaface_cov import RetinaFaceCoV
from api_usage.Load_model_iresnet import Loadmodel
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper


def make_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def record(x, y, output_path):
    save = output_path + '/' + str(datetime.datetime.now().year) + '/' + str(datetime.datetime.now().month) + '/' + str(
        datetime.datetime.now().day)
    make_dir(save)
    subdir = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    original = save + '/' + "original_%s.avi" % (subdir)
    return cv2.VideoWriter(original, fourcc, 24, (x, y))


def main(args):
    mode = args.mode
    if mode == "camera":
        camera_recog()
    elif mode == "input":
        create_manual_data()
    else:
        raise ValueError("Unimplemented mode")


# TODO: DETECT
# define detector
def detect(img, detector):
    thresh = 0.8
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


# TODO: Face recognition process

def open_dataset(data_file):
    f = open(data_file, 'r')
    data_set = json.loads(f.read())
    zeros = list(np.zeros(512))
    axis_max = 0
    dataset = []
    people_names = []
    for person in data_set.keys():
        person_data = data_set[person]
        person_data = np.array(person_data)
        number_vector = person_data.shape[0]
        if number_vector > axis_max:
            axis_max = number_vector

    for person in data_set.keys():
        person_data = data_set[person]
        number_vector = len(person_data)
        if number_vector < axis_max:
            for i in range(axis_max - number_vector):
                person_data.append(zeros)
        dataset.append(person_data)
        people_names.append(person)

    dataset = np.array(dataset)
    dataset = np.transpose(dataset, axes=(0, 2, 1))
    return dataset, people_names


def camera_recog():
    '''
    Description:
    Images from Video Capture -> detect faces' regions -> crop those faces and align them
        -> each cropped face is categorized in 3 types: Center, Left, Right
        -> Extract 512D vectors( face features)
        -> Search for matching subjects in the dataset based on the types of face positions.
        -> The preexisitng face 512D vector with the shortest distance to the 512D vector of the face on screen is most likely a match
        (Distance threshold is 0.6, percentage threshold is 70%)
    '''

    # Load dataset
    thres = 0.7
    data_file = './VTS_236.txt'
    print("Load dataset...")
    # dataset, people_names = open_dataset(data_file)
    f = open(data_file, 'r')
    dataset = json.loads(f.read())
    people_names = []

    print("[INFO] camera sensor warming up...")
    # vs = cv2.VideoCapture(0)
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # vs.set(cv2.CAP_PROP_FOURCC, fourcc)
    # vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # vs = cv2.VideoCapture("rtsp://admin:Viettel2020@172.16.10.96:554/Streaming/Channels/101")
    # vs = cv2.VideoCapture("rtsp://admin:Admin123@172.16.10.92:554/Streaming/Channels/101")
    # vs = cv2.VideoCapture("/home/ubuntu/Downloads/test_mask/vmsteam.mp4")
    vs = cv2.VideoCapture("/media/ubuntu/DATA_QUANG/maythanh/SPOOFING_FACE_ORI/videos/2020/2/12/original_20200212-112100.avi")

    # out_done = record(1920, 1080, './video_output')

    print("Load model...")
    gpuid = 0
    detector = RetinaFaceCoV('retina/model/cov2/mnet_cov2', 0, gpuid, 'net3l')
    faceRecModelHandler = Loadmodel()
    face_cropper = FaceRecImageCropper()

    while True:
        success, image = vs.read()
        if success != True:
            break
        if image is None:
            continue
        # image = cv2.flip(image, 1)
        bounding_boxes, landmarks_fivepoints = detect(image, detector)

        features_arr = []
        if len(bounding_boxes) > 0:
            for i in range(0, len(bounding_boxes)):
                landmarks = landmarks_fivepoints[i]
                cropped_image = face_cropper.crop_image_by_mat(image, landmarks)    # BGR
                feature = faceRecModelHandler.inference_on_image(cropped_image)
                features_arr.append(feature)

            for i in range(0, len(bounding_boxes)):
                det = bounding_boxes[i]
                recog_data = findPeople(features_arr, dataset, people_names, thres)
                # print("time for find people in db is %.5f" % (time() - start_time))
                score = str(format(recog_data[i][1], '.2f'))
                if recog_data[i][0] == "Unknown":
                    cv2.rectangle(image, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 0, 255),
                                  3)  # draw bounding box for the face
                    cv2.putText(image, recog_data[i][0] + '_' + str(recog_data[i][1]), (int(det[0]), int(det[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(image, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (255, 200, 0),
                                  3)  # draw bounding box for the face
                    cv2.putText(image, recog_data[i][0] + '_' + str(recog_data[i][1]), (int(det[0]), int(det[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 200, 0), 2, cv2.LINE_AA)
                    print("-------------%s:%s" % (recog_data[i][0], score))

        cv2.namedWindow('Demo mask', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Demo mask', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Demo mask", image)
        # out_done.write(image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def findPeople(features_arr, dataset, people_names, thres=0.55, percent_thres=80):
    '''
    facerec_512D.txt Data Structure:
    {
    "Person ID": {
        [[512D vector],
        [512D vector],
        [512D Vector]]
        }
    }
    This function basically does a simple linear search for
    ^the 512D vector with the min distance to the 512D vector of the face on screen
    '''
    '''
    :param features_arr: a list of 512d Features of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''

    returnRes = []

    for (i, features_512D) in enumerate(features_arr):
        result = "Unknown"
        smallest = sys.maxsize
        for person in dataset.keys():
            person_data = dataset[person]
            person_data = np.array(person_data)
            for j in range(person_data.shape[0]):
                distance = np.sqrt(np.sum(np.square(person_data[j] - features_512D)))
                if (distance < smallest):
                    smallest = distance
                    result = person
        percentage = min(100, 100 * thres / smallest)
        if percentage <= percent_thres:
            result = "Unknown"
        returnRes.append((result, round(percentage, 2)))

    return returnRes


# TODO: Face registration
def getPos(points):
    if abs(points[0] - points[2]) / abs(points[1] - points[2]) > 2:
        return "Right"
    elif abs(points[1] - points[2]) / abs(points[0] - points[2]) > 2:
        return "Left"
    return "Center"


def create_manual_data():
    '''
    Description:
    User input his/her name or ID -> Images from Video Capture -> detect the face -> crop the face and align it
        -> face is then categorized in 3 types: Center, Left, Right
        -> Extract 512D vectors( face features)
        -> Append each newly extracted face 512D vector to its corresponding position type (Center, Left, Right)
        -> Press Q to stop capturing
        -> Find the center ( the mean) of those 512D vectors in each category. ( np.mean(...) )
        -> Save
    '''

    vs = cv2.VideoCapture(0)
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # vs = cv2.VideoCapture("rtsp://admin:Viettel2020@172.16.10.96:554/Streaming/Channels/101")
    # vs = cv2.VideoCapture("rtsp://admin:Admin123@172.16.10.92:554/Streaming/Channels/101")

    print("Please input new user ID:")
    new_name = input()  # ez python input()

    print("Load model...")
    gpuid = 0
    detector = RetinaFaceCoV('retina/model/cov2/mnet_cov2', 0, gpuid, 'net3l')
    faceRecModelHandler = Loadmodel()
    face_cropper = FaceRecImageCropper()

    txt_file = "./VTSmask_512D.txt"

    f = open(txt_file, 'r')
    data_set = json.loads(f.read())
    positions_arr = []
    features_arr = []

    print("Please start turning slowly. Press 'q' to save and add this new user to the dataset")
    while True:
        success, image = vs.read()
        if success != True:
            print("Error camera connection")
            break
        if image is None:
            print("Error camera connection")
            continue
        image = cv2.flip(image, 1)
        bounding_boxes, landmarks_fivepoints = detect(image, detector)

        if len(bounding_boxes) == 0:
            print("Let try!")
        elif len(bounding_boxes) > 1:
            print("More than 1 person")
        elif len(bounding_boxes) == 1:
            for (i, det) in enumerate(bounding_boxes):
                position = getPos(landmarks_fivepoints[i, :, 0])
                if position in positions_arr:
                    print("Continue turn please")
                else:
                    positions_arr.append(position)
                    landmarks = landmarks_fivepoints[i]
                    cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
                    feature = faceRecModelHandler.inference_on_image(cropped_image)
                    features_arr.append(feature)
                if len(positions_arr) == 3:
                    features_arr = np.array(features_arr).tolist()
                    data_set[new_name] = features_arr
                    f = open(txt_file, 'w')
                    f.write(json.dumps(data_set))
                    sys.exit(-1)
                cv2.rectangle(image, (det[0], det[1]), (det[2], det[3]), (0, 255, 255),
                              2)  # draw bounding box for the face
                cv2.putText(image, position, (det[0], det[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.namedWindow('Captured face', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Captured face', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Captured face", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera")
    args = parser.parse_args(sys.argv[1:])

    main(args)
