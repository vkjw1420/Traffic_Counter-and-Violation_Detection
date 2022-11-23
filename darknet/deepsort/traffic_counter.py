import os
import sys

from numpy.lib.function_base import average
# 상위 디렉토리 패키지 가져오기
p = os.path.abspath('..')
sys.path.insert(1, p)

import numpy as np
import matplotlib.pyplot as plt
import time

# image resizing
import imutils

# for yolov4 detection
from ctypes import *
import random
import cv2
import time
import darknet
import argparse
from threading import Thread
from queue import Queue
# for DeepSORT
import core.utils as utils
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import threading

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="./data/왼쪽_short.mp4",
                        help="video source. If empty, uses ./data/왼쪽_short.mp4")
    parser.add_argument("--output", type=str, default="",
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
    parser.add_argument("--size", type=int, default=608,
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


# 마우스 드래깅으로 ROI 설정 시도
def onMouse(event,x,y,flags,param):     # 마우스 이벤트 핸들 함수  ---①
    global isDragging, x0, y0      # 전역변수 참조
    img = param
    blue, red = (255,0,0),(0,0,255)         # 색상 값 
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 버튼 다운, 드래그 시작 ---②
        isDragging = True
        x0 = x
        y0 = y
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 움직임 ---③
        if isDragging:                  # 드래그 진행 중
            img_draw = img.copy()       # 사각형 그림 표현을 위한 이미지 복제
            cv2.rectangle(img_draw, (x0, y0), (x, y), blue, 2) # 드래그 진행 영역 표시
            cv2.imshow('img', img_draw) # 사각형 표시된 그림 화면 출력
    elif event == cv2.EVENT_LBUTTONUP:  # 왼쪽 마우스 버튼 업 ---④
        if isDragging:                  # 드래그 중지
            isDragging = False          
            
            w = x - x0                  # 드래그 영역 폭 계산
            h = y - y0                  # 드래그 영역 높이 계산
            roi_pos.append(int(x0))
            roi_pos.append(int(y0))
            roi_pos.append(int(w))
            roi_pos.append(int(h))
            print("x:%d, y:%d, w:%d, h:%d" % (x0, y0, w, h))
            if w > 0 and h > 0:         # 폭과 높이가 양수이면 드래그 방향이 옳음 ---⑤
                img_draw = img.copy()   # 선택 영역에 사각형 그림을 표시할 이미지 복제
                # 선택 영역에 빨간 사각형 표시
                cv2.rectangle(img_draw, (x0, y0), (x, y), red, 2) 
                cv2.imshow('img', img_draw) # 빨간 사각형 그려진 이미지 화면 출력
                roi = img[y0:y0+h, x0:x0+w] # 원본 이미지에서 선택 영영만 ROI로 지정 ---⑥
                cv2.imshow('cropped', roi)  # ROI 지정 영역을 새창으로 표시
                print("croped.")
            else:
                cv2.imshow('img', img)  # 드래그 방향이 잘못된 경우 사각형 그림이 없는 원본 이미지 출력
                print("좌측 상단에서 우측 하단으로 영역을 드래그 하세요.")

# 마우스 드래깅으로 ROI 설정 시도
def setROI(img):
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', onMouse, img) # 마우스 이벤트 등록 ---⑧
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    return True


def video_capture(frame_queue, darknet_image_queue):
    isROIset = False
    while cap.isOpened():
        ret, frame = cap.read()
        if isROIset is False:
            img = frame
            # setROI(img)
            isROIset = True
        if not ret:
            break

        # 관심영역 추출.
        roi = frame[ROI_ymin:ROI_ymax, ROI_xmin:ROI_xmax] 
        roi = cv2.fillConvexPoly(roi, nonROI, (0,0,0))
        ROI_frame = imutils.resize(roi.copy(), width = int((ROI_xmax - ROI_xmin)/resizing_rate))

        frame_rgb = cv2.cvtColor(ROI_frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        images = (img_for_detect, ROI_frame)
        darknet_image_queue.put(images)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue, frame_num_queue, cap):
    max_fps = 0
    min_fps = 10000
    total_fps = 0.0
    average_fps = 0.0
    xmax = 0.0
    ymax = 0.0
    frame_num = 0
    while cap.isOpened():
        darknet_image, frame = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        
        detections_queue.put(detections)
        darknet.free_image(darknet_image)
        
        # 여기서 딥소팅을 하자.
        bboxes = []
        scores = []

        # yolov4의 BBOX, confidence 값 추출
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
        # allowed_classes = ['PM', 'car', 'cargo', 'bus', 'bike']

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

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        sort_detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        
        # run non-maxima supression
        boxs = np.array([d.tlwh for d in sort_detections])
        for i in range(0,len(boxs)):
            if xmax < boxs[i][0] + boxs[i][2]:
                xmax = boxs[i][0] + boxs[i][2]
                # print(str(boxs[i][0]) + '    ' + str(boxs[i][1]) + '    ' + str(boxs[i][2]) + '    ' + str(boxs[i][3]))
                boxs[i][2] = video_width - boxs[i][0]
            if ymax < boxs[i][1] + boxs[i][3]:
                ymax = boxs[i][1] + boxs[i][3]

        # print("xmax : " + str(xmax) + "    ymax : " + str(ymax))

        scores = np.array([d.confidence for d in sort_detections])
        classes = np.array([d.class_name for d in sort_detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        sort_detections = [sort_detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(sort_detections, video_height)

        tracker_queue.put(tracker)
        bboxes_queue.put(boxs)

        # frame rate
        frame_num +=1
        fps = int(1/(time.time() - prev_time))
        if max_fps < fps:
            max_fps = fps
        if min_fps > fps:
            min_fps = fps
        fps_queue.put(fps)
        total_fps += float(fps)
        frame_num_queue.put(frame_num)
        average_fps = total_fps/frame_num
        print("Frame #: {}      FPS: {}      AVG_FPS: {:.3f}        MAX_FPS:  {}        MIN_FPS:  {}"
                .format(frame_num, fps, average_fps, max_fps, min_fps))
        
        # # if enable info flag then print details about each track
        #   if FLAGS.info:
        #    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        
        
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue, frame_num_queue):
    # traffic tracking용 변수
    trafficOut = 0
    trafficIn = 0

    lane1In = 0
    lane2In = 0
    lane3In = 0
    lane4In = 0
    lane5In = 0

    lane1Speed = 0
    lane2Speed = 0
    lane3Speed = 0
    lane4Speed = 0
    lane5Speed = 0

    totalLane1Speed = 0
    totalLane2Speed = 0
    totalLane3Speed = 0
    totalLane4Speed = 0
    totalLane5Speed = 0

    avglane1Speed = 0
    avglane2Speed = 0
    avglane3Speed = 0
    avglane4Speed = 0
    avglane5Speed = 0

    lane1Out = 0
    lane2Out = 0
    lane3Out = 0
    lane4Out = 0

    PM_in = 0
    car_in = 0
    bus_in = 0
    cargo_in = 0
    bike_in = 0

    PM_out = 0
    car_out = 0
    bus_out = 0
    cargo_out = 0
    bike_out = 0
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.output, (video_width, video_height))
    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        frame_num = frame_num_queue.get()
        tracker = tracker_queue.get()
        bboxes = bboxes_queue.get()

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        if frame is not None:

            # yolo bbox drawing
            # detections_adjusted = []
            # for label, confidence, bbox in detections:
            #         bbox_adjusted = convert2original(frame, bbox)
            #         detections_adjusted.append((str(label), confidence, bbox_adjusted))
            #         frame = darknet.draw_boxes(detections_adjusted, frame, class_colors)

            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                x1, y1, x2, y2 = bbox

                x1 *= resizing_rate; x2 *= resizing_rate
                y1 *= resizing_rate; y2 *= resizing_rate
                x1 += ROI_xmin;      x2 += ROI_xmin
                y1 += ROI_ymin;      y2 += ROI_ymin
                
                w = x2 - x1
                h = y2 - y1

                bbox = [int(round(x1 - (w / 2))), int(round(y1 - (h / 2))), 
                        int(round(x2 - (w / 2))), int(round(y2 - (h / 2)))]
                class_name = track.get_class()

                c1 = (int(bbox[0]) + int(bbox[2]))/2
                c2 = (int(bbox[1]) + int(bbox[3]))/2
                centerPoint = (int(c1), int(c2))

                # traffic count
                # 좌측 객체
                if track.stateOutMetro == 1 and (750 < int(bbox[1]+h/1.5)) and track.noConsider == False:######
                        trafficIn += 1

                        # 각 차선에 대한 좌표 값.
                        if 412 <= centerPoint[0] <= 533:
                            lane5In += 1
                        if 533 <= centerPoint[0] <= 697:
                            lane4In += 1 
                        if 697 <= centerPoint[0] <= 877:
                            lane3In += 1
                        if 877 <= centerPoint[0] <= 1050:
                            lane2In += 1
                        if 1050 <= centerPoint[0] <= 1209:
                            lane1In += 1

                        if track.has_speed is True:
                            if 412 <= centerPoint[0] <= 533:
                                lane5Speed = track.speed
                                if  lane5In != 0:
                                    totalLane5Speed += lane5Speed
                                    avglane5Speed = totalLane5Speed/lane5In
                            if 533 <= centerPoint[0] <= 697:
                                lane4Speed = track.speed
                                if lane4In != 0:
                                    totalLane4Speed += lane4Speed
                                    avglane4Speed = totalLane4Speed/lane4In
                            if 697 <= centerPoint[0] <= 877:
                                lane3Speed = track.speed
                                if lane3In != 0:
                                    totalLane3Speed += lane3Speed
                                    avglane3Speed = totalLane3Speed/lane3In
                            if 877 <= centerPoint[0] <= 1050:
                                lane2Speed = track.speed
                                if lane2In != 0:
                                    totalLane2Speed += lane2Speed
                                    avglane2Speed = totalLane2Speed/lane2In
                            if 1050 <= centerPoint[0] <= 1209:
                                lane1Speed = track.speed
                                if lane1In != 0:
                                    totalLane1Speed += lane1Speed
                                    avglane1Speed = totalLane1Speed/lane1In

                        if class_name == 'car':
                            car_in += 1
                        elif class_name == 'cargo':
                            cargo_in += 1
                        elif class_name == 'bus':
                            bus_in += 1
                        elif class_name == 'bike':
                            bike_in += 1
                        elif class_name == 'PM':
                            PM_in += 1
                        
                        track.noConsider = True
                        cv2.line(frame, (0, video_height // 2 -40), (455, video_height // 2 -40), (0, 255, 0), 2)
                
                # 실질적으로 카운트 되는 선, 원래 값-(H/2 +50) y축: 중앙 + 50
                # 우측 객체
                # if track.stateOutMetro == 0 and (460 > int(bbox[1]+h/1.5)) and track.noConsider == False: 
                #     trafficOut += 1

                #     # 각 차선에 대한 좌표 값.
                #     if 892 <= centerPoint[0] <= 950:
                #         lane4Out += 1
                #     if 950 <= centerPoint[0] <= 1010:
                #         lane3Out += 1 
                #     if 1010 <= centerPoint[0] <= 1065:
                #         lane2Out += 1
                #     if 1065 <= centerPoint[0] <= 1125:
                #         lane1Out += 1

                #     if class_name == 'car':
                #         car_out += 1
                #     elif class_name == 'cargo':
                #         cargo_out += 1
                #     elif class_name == 'bus':
                #         bus_out += 1
                #     elif class_name == 'bike':
                #         bike_out += 1
                #     elif class_name == 'PM':
                #         PM_out += 1

                #     track.noConsider = True
                #     cv2.line(frame, (455, video_height // 2 - 40), (video_width, video_height // 2 - 40), (255, 0, 0), 2)
                #############################################################################################################

                # # 차선 위치 계산
                # if line_start > int(bbox[1]+h/1.5) and line_end < int(bbox[1]+h/1.5):
                #     lane_number = 0
                #     for line in lines:
                #         # 1차 방정식으로 현재 위치하고 있는 차선을 인식
                #         slope = (line[1][1] - line[0][1])/(line[1][0] - line[0][0])
                #         y_intercept = line[0][1] - slope*line[0][0]
                #         line_x_pos = (int(bbox[1]+h/1.5) - y_intercept)/slope
                #         if line_x_pos < centerPoint[0]:
                #             track.laneNumber = lane_number
                #             lane_number += 1
                #         else:
                #             if track.postLaneNumber is None:
                #                 track.postLaneNumber = track.laneNumber
                #             # 연속 차선 변경 감지
                #             if track.postLaneNumber != track.laneNumber:
                #                 if track.changeLane is False:
                #                     track.changeLane = True
                #                 if track.changeLaneTime == 0:
                #                     track.changeLaneTime = frame_num
                #                 else:
                #                     wait_time = (frame_num - track.changeLaneTime)/frame_rate
                #                     lane_gap = track.postLaneNumber - track.laneNumber
                #                     if wait_time > lane_stay_time:
                #                         track.postLaneNumber = lane_number
                #                         track.changeLane = False
                #                     elif wait_time < lane_stay_time and (lane_gap >= 2 or lane_gap <= -2):
                #                         track.multipleLaneChange = True
                #                         track.postLaneNumber = lane_number
                #                         track.changeLane = False
                #             break

                # calculated speed
                # 좌측 속도 계산
                if track.stateOutMetro == 1 and track.calc_speed == False and (speed_up_y < int(bbox[1]+h/1.5)) and (speed_down_y > int(bbox[1]+h/1.5)):
                    track.calc_speed = True
                    track.first_fps = frame_num
                if track.stateOutMetro == 1 and track.calc_speed == True and (speed_down_y < int(bbox[1]+h/1.5)) and track.has_speed == False:
                    total_fps = frame_num - track.first_fps
                    # 동 단위 도로에서 점선은 3m실선 + 5m빈공간으로 이루어짐. 0.016은 총 2칸의 점선 16m의 km 표기를 의미
                    track.speed = int(0.016/((total_fps / frame_rate)/3600))
                    track.has_speed = True

                # 우측 속도 계산
                # if track.stateOutMetro == 0 and track.calc_speed == False and (speed_down_y > int(bbox[1]+h/1.5)) and (speed_up_y < int(bbox[1]+h/1.5)):
                #     track.calc_speed = True
                #     track.first_fps = frame_num
                # if track.stateOutMetro == 0 and track.calc_speed == True and (speed_up_y > int(bbox[1]+h/1.5)) and track.has_speed == False:
                #     total_fps = frame_num - track.first_fps
                #     track.speed = int(0.016/((total_fps / frame_rate)/3600))
                #     track.has_speed = True
                    
                # if track.speed > speed_limit:
                #         track.speeding = True

                # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                text_color = (255,255,255)
                # # 과속
                # if track.speeding is True:
                #     color = (0,0,255)
                #     text_color = (0,255,255)
                #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                #     cv2.putText(frame, "Speeding", (int(bbox[0]), int(bbox[1]-25)),0, 1, text_color,2)
                # # 연속 차선 변경
                # elif track.multipleLaneChange is True:
                #     color = (0,0,255)
                #     text_color = (0,255,255)
                #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                #     cv2.putText(frame, "MultipleLaneChange", (int(bbox[0]), int(bbox[1]-25)),0, 1, text_color,2)

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-20)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*13, int(bbox[1])+5), color, -1)
                cv2.circle(frame, (int(bbox[0]+w/2), int(bbox[1]+h/1.5)), 4, color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1]-5)),0, 0.5, text_color,1)
                if track.has_speed is True:
                    cv2.putText(frame, " - " + str(track.speed) + "km/h", (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*13, int(bbox[1]-5)),0, 0.5, text_color,1)

                # if not track.laneNumber is None:
                #     if track.has_speed is True:
                #         cv2.putText(frame, " - " + str(track.laneNumber) + " lane", 
                #                     (int(bbox[0])+(len(class_name)+len(str(track.track_id))+len(str(track.speed)+"km/h")+len(str(track.laneNumber)))*13, int(bbox[1]-5)),0, 0.5, text_color,1)
                #     else:
                #         cv2.putText(frame, " - " + str(track.laneNumber) + " lane", (int(bbox[0])+(len(class_name)+len(str(track.laneNumber)))*13, int(bbox[1]-5)),0, 0.5, text_color,1)


            # draw ROI 영역
            cv2.rectangle(frame, (ROI_xmin, ROI_ymin - 60), (ROI_xmin + 110, ROI_ymin), (233,250,50), -1)
            cv2.putText(frame, 'ROI', (ROI_xmin, ROI_ymin - 10), 0, 2, (0,0,0), 2)
            cv2.rectangle(frame, (ROI_xmin, ROI_ymin), (ROI_xmax, ROI_ymax), (233,250,50), 1)

            # # 차선 양 옆 라인 그리기
            # for line in lines:
            #     cv2.line(frame, line[0], line[1], (0,0,0), 4)

            # 차량이 카운트 되는 기준 선 표시
            # In lanes(좌측) 가장 안쪽이 1차선.
            cv2.line(frame, (412, 750), (533, 750), (255, 0, 0), 2) # 5lane
            cv2.line(frame, (533, 750), (697, 750), (0, 255, 0), 2) # 4lane
            cv2.line(frame, (697, 750), (877, 750), (0, 0, 255), 2) # 3lane
            cv2.line(frame, (877, 750), (1050, 750), (255, 0, 255), 2) # 2lane
            cv2.line(frame, (1050, 750), (1209, 750), (255, 255, 0), 2) # 1lane
            # Out lanes(우측) 가장 안쪽이 1차선
            # cv2.line(frame, (892, 460), (950, 460), (255, 0, 0), 2) # 1lane
            # cv2.line(frame, (950, 460), (1010, 460), (0, 255, 0), 2) # 2lane
            # cv2.line(frame, (1010, 460), (1065, 460), (0, 0, 255), 2) # 3lane
            # cv2.line(frame, (1065, 460), (1125, 460), (255, 0, 255), 2) # 4lane
            
            info = [
                # ('traffic Count In', trafficIn),
                ('traffic Count', trafficIn),
                # ('traffic Count Out', trafficOut), 
            ] 

            in_count_info = [lane5In, lane4In, lane3In,lane2In, lane1In]

            in_speed_info = [lane5Speed,lane4Speed,lane3Speed,lane2Speed,lane1Speed]

            in_type_info = [
                ('PM', PM_in),
                ('car', car_in),
                ('cargo', cargo_in),
                ('bus', bus_in),
                ('bike', bike_in)
            ]
            
            # out_count_info = [lane4Out, lane3Out, lane2Out, lane1Out]

            # out_type_info = [
            #     ('PM', PM_out),
            #     ('car', car_out),
            #     ('cargo', cargo_out),
            #     ('bus', bus_out),
            #     ('bike', bike_out)
            # ]

            # In 차선 카운터 표시(좌측 하단)
            # cv2.rectangle(frame, (0, video_height - 70), (550, video_height), (255,255,255), -1) # 표시 부분(아래)
            # cv2.line(frame, (110,video_height - 70), (110,video_height), (0,0,0), 1)
            # cv2.line(frame, (220,video_height - 70), (220,video_height), (0,0,0), 1)
            # cv2.line(frame, (330,video_height - 70), (330,video_height), (0,0,0), 1)
            # cv2.line(frame, (440,video_height - 70), (440,video_height), (0,0,0), 1)
            # In 차선 카운트 값
            cv2.putText(frame, str(in_count_info[0]), (412 + 10,800), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
            cv2.putText(frame, str(in_count_info[1]), (533 + 10,800), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.putText(frame, str(in_count_info[2]), (697 + 10,800), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.putText(frame, str(in_count_info[3]), (877 + 10,800), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 2)
            cv2.putText(frame, str(in_count_info[4]), (1050 + 10,800), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
            # cv2.putText(frame, str(in_count_info[0]), (45,video_height - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
            # cv2.putText(frame, str(in_count_info[1]), (155,video_height - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            # cv2.putText(frame, str(in_count_info[2]), (265,video_height - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            # cv2.putText(frame, str(in_count_info[3]), (375,video_height - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 2)
            # cv2.putText(frame, str(in_count_info[4]), (485,video_height - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

            # average speed
            cv2.putText(frame, "{:.1f}".format(avglane5Speed) + "km/h", (412 + 10,830), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
            cv2.putText(frame, "{:.1f}".format(avglane4Speed) + "km/h", (533 + 10,830), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.putText(frame, "{:.1f}".format(avglane3Speed) + "km/h", (697 + 10,830), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.putText(frame, "{:.1f}".format(avglane2Speed) + "km/h", (877 + 10,830), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 2)
            cv2.putText(frame, "{:.1f}".format(avglane1Speed) + "km/h", (1050 + 10,830), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

            # Out 차선 카운터 표시(우측 상단)
            # cv2.rectangle(frame, (video_width - 440, 0), (video_width, 45), (255,255,255), -1) # 표시 부분(위)
            # cv2.line(frame, (video_width - 110,0), (video_width - 110,45), (0,0,0), 1)
            # cv2.line(frame, (video_width - 220,0), (video_width - 220,45), (0,0,0), 1)
            # cv2.line(frame, (video_width - 330,0), (video_width - 330,45), (0,0,0), 1)
            # Out 차선 카운트 값
            # cv2.putText(frame, str(out_count_info[3]), (video_width - 65,32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 2)
            # cv2.putText(frame, str(out_count_info[2]), (video_width - 175,32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            # cv2.putText(frame, str(out_count_info[1]), (video_width - 285,32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            # cv2.putText(frame, str(out_count_info[0]), (video_width - 395,32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
            
            # In lanes, Out lanes 통행량
            for( i, (k, v)) in enumerate(info): #add code 
                text = "{}: {}".format(k, v) #add code
                cv2.putText(frame, text, (10, video_height - ((i * 30) + 600)), # (i * x) x부분을 변경하면 글자 높이 간격을 조절할 수 있음
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # In lane 통행 차종
            for( i, (k, v)) in enumerate(in_type_info): #add code 
                text = "{}: {}".format(k, v) #add code
                cv2.putText(frame, text, (100, video_height - ((i * 30) + 100)), # (i * x) x부분을 변경하면 글자 높이 간격을 조절할 수 있음
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Out lane 통행 차종
            # for( i, (k, v)) in enumerate(out_type_info): #add code[]
            #     text = "{}: {}".format(k, v) #add code
            #     cv2.putText(frame, text, (video_width - 170, video_height - ((i * 30) + 600)), # (i * x) x부분을 변경하면 글자 높이를 간격을 조절할 수 있음
            #     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.putText(frame, 'fps : ' + str(fps), (10,50), 0, 2, (0,255,0), 3)

            if video_height >= 720 and video_width >= 1280:
                show = cv2.resize(frame, (1280, 720))
            else:
                show = frame
            if not args.dont_show:
                cv2.imshow('Inference', show)
            if args.output is not None:
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
        #     if args.output is not None:
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
    frame_num_queue = Queue(maxsize=1)

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
    # darknet_width = 416
    # darknet_height = 416

    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 마우스 드래깅으로 ROI 설정 시도
    isDragging = False                      # 마우스 드래그 상태 저장 
    isROIset = False                        # ROI 설정 여부 확인
    x0, y0, w, h = -1,-1,-1,-1              # 영역 선택 좌표 저장
    roi_pos = []

    # ROI 영역 수동 지정
    ROI_xmin = 330; ROI_xmax = 1300; ROI_ymin = 350; ROI_ymax = 870  # 구미
    # ROI_xmin = 270; ROI_xmax = 1300; ROI_ymin = 50; ROI_ymax = 870 # 대구
    # ROI_xmin = 0; ROI_xmax = video_width; ROI_ymin = 0; ROI_ymax = video_height  # 전체 이미지

    # 비 관심 영역, 검은색으로 칠해버림
    # 구미
    nonROI = np.array([[562, 0], [562, 110], [630, 167], [715, 167], [970, 386], [970, 0]], np.int32)
    # 대구
    # nonROI = np.array([[562, 0], [562, 110], [630, 167], [715, 167], [970, 386], [970, 0]], np.int32)

    # 영상 축소 수치. ex) 2 -> 1/2. 50% 축소
    resizing_rate = 2

    # 영상 재생 빈도. 영상의 종류에 따라 변경해주어야한다.
    frame_rate = 60

    # 속도 계산을 위한 y 좌표 기준선.line_start
    speed_up_y = 487
    speed_down_y = 540

    frame_num = 0

    # # 속도 제한. 테스트를 위해 허용 속도가 아닌 제한속도를 바로 적용함.
    # speed_limit = 50

    # # 차선들
    # lines = (((300, 870), (625, 460)), ((450, 870), (695, 460)), ((668, 870), (765, 460)), ((894, 870), (825, 460)), ((1120, 870), (892, 460)), 
    #                 ((1315, 870), (950, 460)), ((1552, 870), (1010, 460)), ((1762, 870), (1065, 460)), ((1915, 840), (1125, 460)))
    # line_start = 0
    # line_end = 0
    # for line in lines:
    #     if line[0][1] > line_start:
    #         line_start = line[0][1]
    #     if line[1][1] < line_end:
    #         line_end = line[1][1]

    # # 연속 차선 변경이 되지 않기 위해 차선에 머물러야하는 시간
    # lane_stay_time = 3.0

    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue, frame_num_queue, cap)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue, frame_num_queue)).start()
