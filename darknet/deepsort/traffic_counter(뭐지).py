import time
import os
#from database import * # add
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-4class_1696_416/',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', 'data/video/test.mp4', 'path to input video or set to 0 for webcam')
# flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
#flags.DEFINE_string('output', '/home/ddd/test.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    trafficOut = 0
    trafficIn = 0

    lane1In = 0
    lane2In = 0
    lane3In = 0
    lane4In = 0

    lane1Out = 0
    lane2Out = 0
    lane3Out = 0

    car_in = 0
    truck_in = 0
    bus_in = 0
    bike_in = 0

    car_out = 0
    truck_out = 0
    bus_out = 0
    bike_out = 0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    
    except:
        vid = cv2.VideoCapture(video_path)
       
    out = None
    

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    W = None #add code 
    H = None #add code
    

    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            if W is None or H is None: # add code
                (H, W) = frame.shape[:2] # add code
        else:
            print('Video has ended or failed, try a different video format!')
            #change_total(peopleIn) # about DB
            break
        frame_num +=1
        #print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        # allowed_classes = ['person,']
        #allowed_classes = ['car']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        # H 값을 넘겨주고 그거 활용해서 차량이 좌측인지 우측인지 판단. IN or OUT
        tracker.update(detections, H)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            # class_name = track.get_class() # 주석 처리 했음
            #####################################
            c1 = (int(bbox[0]) + int(bbox[2]))/2 #edit - add 'global' 
            c2 = (int(bbox[1]) + int(bbox[3]))/2 #edit - add 'global'
            centerPoint = (int(c1), int(c2))

            """     bbox[0] = minX
                    bbox[1] = minY 
                    bbox[2] = maxX 
                    bbox[3] = maxY
            """



            # test code
#            a = int(bbox[0])
#            b = int(bbox[1])
#            c = int(bbox[2])
#            d = int(bbox[3])
#            cv2.circle(frame, (a,b), 4, (255,0,0), -1)
#            cv2.circle(frame, (c,d), 4, (0,255,0), -1)
#            cv2.circle(frame, (int(c2),int(c2)), 4, (43,255,54), -1)
            # print(H/2 +50 - (int(bbox[3]) + int(bbox[1])))
            # test code
            
            #############################################################################################################
            cv2.putText(frame, str(track.track_id),centerPoint,0, 5e-3 * 200, (0,0,255),2)
            cv2.circle(frame, centerPoint, 4, (0, 0, 255), -1)
            #print(track.track_id, track.stateOutMetro)

            # noConsider : 카운트 되고 나면 True로 변경. 재카운트 방지
            # 좌측 객체
            if track.stateOutMetro == 1 and (H/2 -40 - (int(bbox[3]) + int(bbox[1]))/2 < 0) and track.noConsider == False:######
                    trafficIn += 1 

                    # 각 차선에 대한 좌표 값.
                    if 0 <= centerPoint[0] <=70:
                        lane4In += 1
                    if 80 <= centerPoint[0] <=200:
                        lane3In += 1 
                    if 230 <= centerPoint[0] <=325:
                        lane2In += 1
                    if 335 <= centerPoint[0] <= 440:
                        lane1In += 1

                    if class_name == 'car':
                        car_in += 1
                    elif class_name == 'truck':
                        truck_in += 1
                    elif class_name == 'bus':
                        bus_in += 1
                    elif class_name == 'bike':
                        bike_in += 1
                    
                    track.stateOutMetro = 0
                    track.noConsider = True
                    cv2.line(frame, (0, H // 2 -40), (455, H // 2 -40), (0, 255, 0), 2)
               
                # 실질적으로 카운트 되는 선, 원래 값-(H/2 +50) y축: 중앙 + 50
            # 우측 객체
            if track.stateOutMetro == 0 and (H/2 -40 - (int(bbox[3]) + int(bbox[1]))/2 >= 0) and track.noConsider == False: 
                trafficOut += 1

                # 각 차선에 대한 좌표 값.
                if 465 <= centerPoint[0] <=575:
                    lane1Out += 1
                if 590 <= centerPoint[0] <=710:
                    lane2Out += 1 
                if 715 <= centerPoint[0] <=830:
                    lane3Out += 1

                if class_name == 'car':
                    car_out += 1
                elif class_name == 'truck':
                    truck_out += 1
                elif class_name == 'bus':
                    bus_out += 1
                elif class_name == 'bike':
                    bike_out += 1

                
                track.stateOutMetro = 1
                track.noConsider = True
                cv2.line(frame, (455, H // 2 - 40), (W, H // 2 - 40), (255, 0, 0), 2)
            #############################################################################################################


        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        
        #cv2.line(frame, (0, H // 2 +50), (W, H // 2 +50), (255, 0, 0), 2) #add code(add blue line) # main lane


        # 표시되는 선
        cv2.line(frame, (0, 200), (70, 200), (255, 255, 0), 2) # 1lane
        cv2.line(frame, (80, 200), (200, 200), (0, 0, 255), 2) # 2lane
        cv2.line(frame, (215, 200), (325, 200), (0, 255, 0), 2) # 3lane
        cv2.line(frame, (335, 200), (440, 200), (255, 0, 0), 2) # 3lane

        cv2.line(frame, (465, 200), (575, 200), (255, 0, 0), 2) # 3lane
        cv2.line(frame, (590, 200), (710, 200), (0, 255, 0), 2) # 3lane
        cv2.line(frame, (715, 200), (830, 200), (0, 0, 255), 2) # 3lane
        
        info = [  #add code
            ('traffic Count In', trafficIn), #add code
            ('traffic Count Out', trafficOut), #add code
        ] #add code

        in_count_info = [lane4In, lane3In,lane2In, lane1In]

        in_type_info = [
            ('car', car_in),
            ('truck', truck_in),
            ('bus', bus_in),
            ('bike', bike_in)
        ]
        
        out_count_info = [lane3Out, lane2Out, lane1Out]

        out_type_info = [
            ('car', car_out),
            ('truck', truck_out),
            ('bus', bus_out),
            ('bike', bike_out)
        ]

        cv2.rectangle(frame, (0, 435), (455, H), (255,255,255), -1) # 표시 부분(아래)
        cv2.line(frame, (113,435), (113,H), (0,0,0), 1)
        cv2.line(frame, (226,435), (226,H), (0,0,0), 1)
        cv2.line(frame, (339,435), (339,H), (0,0,0), 1)
        
        # for(i, k) in enumerate(in_count_info): # int_count_info 차선 부분
        #     text = "{}".format(k) #add code
        #     cv2.putText(frame, text, (i * 250, H - ((i * 0) + 10)), # (i * x) x부분을 변경하면 글자 높이 간격을 조절할 수 있음
        #     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.putText(frame, str(in_count_info[0]), (50,465), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
        cv2.putText(frame, str(in_count_info[1]), (160,465), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.putText(frame, str(in_count_info[2]), (270,465), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, str(in_count_info[3]), (380,465), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

        
        cv2.rectangle(frame, (450, 0), (W, 45), (255,255,255), -1) # 표시 부분(위)
        cv2.line(frame, (584,0), (584,45), (0,0,0), 1)
        cv2.line(frame, (718,0), (718,45), (0,0,0), 1)
        
        cv2.putText(frame, str(out_count_info[2]), (510,32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
        cv2.putText(frame, str(out_count_info[1]), (645,32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, str(out_count_info[0]), (780,32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        
        for( i, (k, v)) in enumerate(info): #add code 
            text = "{}: {}".format(k, v) #add code
            cv2.putText(frame, text, (10, H - ((i * 30) + 420)), # (i * x) x부분을 변경하면 글자 높이 간격을 조절할 수 있음
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        for( i, (k, v)) in enumerate(in_type_info): #add code 
            text = "{}: {}".format(k, v) #add code
            cv2.putText(frame, text, (10, H - ((i * 30) + 51)), # (i * x) x부분을 변경하면 글자 높이 간격을 조절할 수 있음
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        for( i, (k, v)) in enumerate(out_type_info): #add code[]
            text = "{}: {}".format(k, v) #add code
            cv2.putText(frame, text, (720, H - ((i * 30) + 315)), # (i * x) x부분을 변경하면 글자 높이를 간격을 조절할 수 있음
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        #insert_database(peopleIn, trafficOut) # about DB

        # if enable info flag then print details about each track
        if FLAGS.info:
            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        #print(H/2 -40)
        # print(f"Width:{W}, Height:{H}")
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.namedWindow('Output Video', cv2.WINDOW_NORMAL)
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cv2.destroyAllWindows()
    #change_total(peopleIn) # about DB

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass



""" 2020-10-24
    미리 학습된 yolov4.weight 파일을 사용했을 경우에는 평균 fps가 14 정도 나오지만 인식은 확실했다.
    직접 라벨링 해서 뽑아낸 weight파일을 사용했을 경우에는 평균 fps가 23정도는 나오지만 인식이 거의 안되는 모습을 보였다.
"""
