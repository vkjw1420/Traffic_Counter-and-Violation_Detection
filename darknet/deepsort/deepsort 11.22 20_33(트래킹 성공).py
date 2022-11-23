import os
import sys
# 상위 디렉토리 패키지 가져오기
p = os.path.abspath('..')
sys.path.insert(1, p)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# for yolov4 detection
from ctypes import *
import random
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
# for DeepSORT
import core.utils as utils
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="./data/sample1.mp4",
                        help="video source. If empty, uses ./data/sample1.mp4")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. auto save if empty")
    parser.add_argument("--weights", default="./yolo_files/yolov4-my-model-v3_best.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true', default=True,
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./yolo_files/yolov4-my-model-v3.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./yolo_files/my.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.5,
                        help="remove detections with confidence below this value")
    parser.add_argument("--size", type=int, default=416,
                        help="resize images to")  
    parser.add_argument("--count", type=bool, default=False,
                        help="count objects being tracked on screen")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        images = (img_for_detect, frame)
        darknet_image_queue.put(images)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue, cap):
    xmax = 0.0
    ymax = 0.0
    frame_num = 0
    while cap.isOpened():
        darknet_image, frame = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        
        detections_queue.put(detections)

        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        # darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)
        
        # 여기서 딥소팅을 하자.
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_num +=1
        print('Frame #: ', frame_num)
        # image_data = cv2.resize(frame, (input_size, input_size))
        # image_data = image_data / 255.
        # image_data = image_data[np.newaxis, ...].astype(np.float32)
        # start_time = time.time()

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        bboxes = []
        scores = []
        # max width 608, max height 608
        # for label, confidence, bbox in detections:
        #     scores.append(confidence)
        #     x, y, w, h = bbox
        #     x = x/darknet_width
        #     y = y/darknet_height
        #     w = w/darknet_width
        #     h = h/darknet_height
        #     bboxes.append([x, y, w, h])

        for label, confidence, bbox in detections:
            scores.append(confidence)
            bboxes.append(convert2original(frame, bbox))

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        # original_h, original_w, _ = frame.shape
        # print(frame.shape)
        # bboxes = utils.format_yolov4_boxes(bboxes, original_h, original_w)

        # by default allow all classes in .names file
        allowed_classes = class_names
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(len(detections)):
            class_name = detections[i][0]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        # detected objects count & show
        if args.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        
        # delete detections that are not in allowed_classes
        # bboxes = np.delete(bboxes, deleted_indx, axis=0)
        # scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        sort_detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        
        # run non-maxima supression
        boxs = np.array([d.tlwh for d in sort_detections])
        for i in range(0,len(boxs)):
            if xmax < boxs[i][0] + boxs[i][2]:
                xmax = boxs[i][0] + boxs[i][2]
            if ymax < boxs[i][1] + boxs[i][3]:
                ymax = boxs[i][1] + boxs[i][3]

        print("xmax : " + str(xmax) + "    ymax : " + str(ymax))

        scores = np.array([d.confidence for d in sort_detections])
        classes = np.array([d.class_name for d in sort_detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        sort_detections = [sort_detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(sort_detections)

        tracker_queue.put(tracker)
        bboxes_queue.put(boxs)
        
        # # update tracks
        # for track in tracker.tracks:
        #     if not track.is_confirmed() or track.time_since_update > 1:
        #         continue 
        #     bbox = track.to_tlbr()
        #     class_name = track.get_class()
            
        # # draw bbox on screen
        #     color = colors[int(track.track_id) % len(colors)]
        #     color = [i * 255 for i in color]
        #     # frame = frame_queue.get()
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        #     print(bbox)
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        #     cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        
        # # if enable info flag then print details about each track
        #     if FLAGS.info:
        #    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        
        # calculate frames per second of running detections
        # fps = 1.0 / (time.time() - start_time)
        #print("FPS: %.2f" % fps)
        #result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # cv2.imshow('output', result)
        
        # if not FLAGS.dont_show:
        #     cv2.imshow("Output Video", result)
        
        # # if output flag is set, save video file
        # if FLAGS.output:
        #     out.write(result)
        
        #if cv2.waitKey(1) & 0xFF == ord('q'): break

        
        
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (video_width, video_height))
    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        tracker = tracker_queue.get()
        bboxes = bboxes_queue.get()
        detections_adjusted = []

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        if frame is not None:
            # frame = cv2.resize(frame, (608, 608))

            # for label, confidence, bbox in detections:
            #         bbox_adjusted = convert2original(frame, bbox)
            #         detections_adjusted.append((str(label), confidence, bbox_adjusted))
            #         image = darknet.draw_boxes(detections_adjusted, frame, class_colors)

            # update tracks
            for track, (label,confidence,bbox) in zip(tracker.tracks, detections):
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1

                bbox = [int(round(x1 - (w / 2))), int(round(y1 - (h / 2))), int(round(x2 - (w / 2))), int(round(y2 - (h / 2)))]
                class_name = track.get_class()
                
                # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                # frame = frame_queue.get()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            if not args.dont_show:
                cv2.imshow('Inference', frame)
            if args.out_filename is not None:
                video.write(frame)
            if cv2.waitKey(fps) == 27:
                break

        # if frame is not None:
        #     for label, confidence, bbox in detections:
        #         bbox_adjusted = convert2original(frame, bbox)
        #         detections_adjusted.append((str(label), confidence, bbox_adjusted))
        #     image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
        #     if not args.dont_show:
        #         cv2.imshow('Inference', image)
        #     if args.out_filename is not None:
        #         video.write(image)
        #     if cv2.waitKey(fps) == 27:
        #         break
    cap.release()
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    nms_max_overlap = 1.0
    max_cosine_distance = 0.4
    nn_budget = None
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    # 트랙커를 통해 물체 정보(bbox, object_id)
    tracker_queue = Queue(maxsize=1)
    # bbox들 을 전달하는 큐
    bboxes_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    input_size = args.size
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue, cap)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()
