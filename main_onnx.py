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
import datetime
import numpy as np
from onnx_utils.features_implement import FacesFeatures

def make_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def create_save_path(save_path):
    name_unknown = os.path.join(save_path, "unknown")
    make_dir(name_unknown)                           # create path to save unknown people
    name_people = os.path.join(save_path, "people")
    make_dir(name_people)                            # create path to save recognize people
    name_videos_path = os.path.join(save_path, "video")
    make_dir(name_videos_path)                       # create path to save processed videos

    return name_unknown, name_people, name_videos_path

def record(x, y, output_path):
    save = output_path + '/' + str(datetime.datetime.now().year) + '/' + str(datetime.datetime.now().month) + '/' + str(
        datetime.datetime.now().day)
    make_dir(save)
    subdir = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    original = save + '/' + "original_%s.avi" % (subdir)
    return cv2.VideoWriter(original, fourcc, 24, (x, y))


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


def main(args, x):
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
    thres_cosin = 0.5
    thres_euclid = 0.78
    percent_thres = 80
    cosin = False

    print("Load dataset...")
    dataset, people_names = open_dataset(args.features_DB)
    # f = open(data_file, 'r')
    # dataset = json.loads(f.read())
    # people_names = []

    print("[INFO] camera sensor warming up...")
    vs = cv2.VideoCapture(args.input_path)
    unknown_path, people_path, _ = create_save_path(args.save_path)

    # out_done = record(1920, 1080, './video_output')

    while True:
        try:
            success, image = vs.read()
        except:
            continue
        if image is None:
            continue

        features_arr, bboxs = x.get_feature(image, one_face=False)
        if len(features_arr) > 0:
            for i in range(0, len(features_arr)):
                det = bboxs[i]
                if cosin:
                    recog_data = findPeople(features_arr, dataset, people_names, thres_cosin, cosin=cosin)
                else:
                    recog_data = findPeople(features_arr, dataset, people_names, thres_euclid, cosin=cosin,
                                            percent_thres=percent_thres)

                score = str(format(recog_data[i][1], '.2f'))

                if args.is_save_images:           # save recognized face image to folder
                    person_name = recog_data[i][0]
                    save_image_name = str(recog_data[i][1]) + '_' + person_name + '_' \
                                      + datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S') + '_' + '.jpg'
                    if recog_data[i][1] < percent_thres:
                        save_image_path = os.path.join(unknown_path, save_image_name)
                    else:
                        save_image_path = os.path.join(people_path, save_image_name)

                    det = np.where(det > 0, det, 1)         # thay the nhung gia tri < 0 trong toa do thanh gia tri bang 1

                    save_img = image[int(det[1]):int(det[3]), int(det[0]):int(det[2])]
                    cv2.imwrite(save_image_path, save_img)

                if recog_data[i][1] < percent_thres:
                    display_name = "Unknown"
                    cv2.rectangle(image, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 0, 255),
                                  3)  # draw bounding box for the face
                    cv2.putText(image, display_name + '_' + str(recog_data[i][1]), (int(det[0]), int(det[1])),
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

def findPeople(features_arr, dataset, people_names, thres=0.5, cosin=True, percent_thres=80):
    '''
    facerec_512D.txt Data Structure:
    {
    "Person ID": {
        [[512D vector],
        [512D vector],
        [512D Vector]]
        }
    }
    This function basically does a simple linear search for the 512D vector with the min distance to the 512D vector of the face on screen
    '''

    '''
    :param features_arr: a list of 512d Features of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''
    returnRes = []
    scores = np.dot(features_arr, dataset)
    scores = np.max(scores, axis=2)
    name_index = np.argmax(scores, axis=1)
    scores = np.max(scores, axis=1)

    for i in range(scores.shape[0]):
        if cosin:
            if scores[i] > thres:
                returnRes.append((people_names[name_index[i]], scores[i]))
            else:
                returnRes.append(("Unknown", scores[i]))
        else:
            euclid_distance = np.sqrt(2-2*scores[i])
            percentage = min(100, 100 * thres / euclid_distance)
            result = people_names[name_index[i]]
            # if percentage <= percent_thres:
            #     result = "Unknown"
            returnRes.append((result, round(percentage, 2)))

    # for (i, features_512D) in enumerate(features_arr):
    #     result = "Unknown"
    #     smallest = sys.maxsize
    #     for person in dataset.keys():
    #         person_data = dataset[person]
    #         person_data = np.array(person_data)
    #         for j in range(person_data.shape[0]):
    #             distance = np.sqrt(np.sum(np.square(person_data[j] - features_512D)))
    #             if (distance < smallest):
    #                 smallest = distance
    #                 result = person
    #     percentage = min(100, 100 * thres / smallest)
    #     if percentage <= percent_thres:
    #         result = "Unknown"
    #     returnRes.append((result, round(percentage, 2)))

        # features_512D = np.expand_dims(features_512D, axis=1)
        # result = "Unknown"
        # biggest = - sys.maxsize
        # for person in dataset.keys():
        #     person_data = dataset[person]
        #     scores = np.dot(person_data, features_512D)
        #     score = np.max(scores)
        #     if score > biggest:
        #         biggest = score
        #         result = person
        # if biggest < 0.5:
        #     result = "Unknown"
        # returnRes.append((result, biggest))

    return returnRes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--recognition_model", type=str, help="recognition model name", default="backbone.onnx")
    parser.add_argument("--detection_model", type=str, help="detection model name", default="model.onnx")
    parser.add_argument("--input_path", type=str, help=" input video path",
                        default="/media/ubuntu/DATA/Trienkhai/haugiang3.mp4")
    parser.add_argument("--features_DB", type=str, help=" text file contains features vector ",
                        default="DB/Haugiang_backbone_mask.txt")
    parser.add_argument("--save_path", type=str, help=" saved images path",
                        default="/media/ubuntu/DATA/Trienkhai/HG-register/recognized_images")
    parser.add_argument("--is_save_images", action="store_false", help="save or not")

    args = parser.parse_args(sys.argv[1:])

    # Load model
    x = FacesFeatures(recognize_model=args.recognition_model, detection_model=args.detection_model, use_gpu=True)
    assets_path = os.path.join('./assets')
    x.load(assets_path)
    main(args, x)
