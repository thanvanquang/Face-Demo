'''
Main program
@Author: Than Van Quang

To execute simply run:
main.py

To input new user:
main.py --mode "input"
'''
import time
import cv2
import argparse
import sys
import os
from onnx_utils.facial_age_expression import FacesFeatures


def main(args, x):
    print("[INFO] camera sensor warming up...")
    vs = cv2.VideoCapture(args.input_path)
    while True:
        t0 = time.time()
        try:
            success, frame = vs.read()
        except:
            continue
        if frame is None:
            continue
        image = cv2.flip(frame, 1)
        bboxs, emotions, agegender = x.run(image)
        for i in range(len(bboxs)):
            det = bboxs[i]
            cv2.rectangle(image, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (255, 200, 0), 1)
            cv2.putText(image, emotions[i][0] + '_' + str(round(emotions[i][1], 2)), (int(det[0]), int(det[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 20, 255), 1, cv2.LINE_AA)  # PutText emotions

            cv2.putText(image, agegender[i][0] + '_' + str(round(agegender[i][2], 2)), (int(det[0]), int(det[3])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 1, cv2.LINE_AA)  # PutText agegender

        cv2.namedWindow('Emotion mask', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Emotion mask', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Emotion mask", image)
        print("time done:", time.time() - t0)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


    # frame = cv2.imread('/home/ubuntu/Desktop/baby2.jpg')
    # image = cv2.flip(frame, 1)
    # bboxs, emotions, agegender = x.run(image)
    # for i in range(len(bboxs)):
    #     det = bboxs[i]
    #     cv2.rectangle(image, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (255, 200, 0), 1)
    #     cv2.putText(image, emotions[i][0] + '_' + str(round(emotions[i][1], 2)), (int(det[0]), int(det[1])),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 20, 255), 1, cv2.LINE_AA)  # PutText emotions
    #
    #     cv2.putText(image, agegender[i][0] + '_' + str(round(agegender[i][2], 2)), (int(det[0]), int(det[3])),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 1, cv2.LINE_AA)  # PutText agegender
    #
    # cv2.namedWindow('Emotion mask', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('Emotion mask', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.imshow("Emotion mask", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agegender_model", type=str, help="recognition model name", default="genderage.onnx")
    parser.add_argument("--emotion_model", type=str, help="recognition model name", default="enet_b2_8")
    parser.add_argument("--detection_model", type=str, help="detection model name", default="model.onnx")
    parser.add_argument("--input_path", type=str, help=" input video path",
                        default=0)

    args = parser.parse_args(sys.argv[1:])

    # Load model
    x = FacesFeatures(detection_model=args.detection_model, agegender_model=args.agegender_model,
                      emotion_model=args.emotion_model, conf_detect_thresh=0.4, num_emotions=8, use_gpu=True)

    assets_path = os.path.join('./assets')
    x.load(assets_path)
    main(args, x)
