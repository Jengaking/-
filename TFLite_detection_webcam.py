######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.
# ###################################################################################################
# 
# edtied by Wonjun, LEE
# project : Crosswalk, 2024/03 - 2024/06
# 
# 

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import threading
import importlib.util
from collections import deque

## IPC mechanism = semaphore
semaphore = 0
pipe = deque()

## var for boundary setting.
click_cnt = 0

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True


#################################################################
################# Object detetion controller ####################
#################################################################
class DetectionController :
    
    def __init__(self) :
        #######################################################################
        ########################  Initialization ##############################
        #######################################################################
        # Define and parse input arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
        parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
        parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

        args = parser.parse_args()

        self.MODEL_NAME = args.modeldir
        self.GRAPH_NAME = args.graph
        self.LABELMAP_NAME = args.labels
        self.min_conf_threshold = float(args.threshold)
        self.resW, self.resH = args.resolution.split('x')
        self.imW, self.imH = int(self.resW), int(self.resH)
        self.use_TPU = args.edgetpu

        # Import TensorFlow libraries
        #   If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
        # If using Coral Edge TPU, import the load_delegate library
        self.pkg = importlib.util.find_spec('tflite_runtime')
        if self.pkg:
            from tflite_runtime.interpreter import Interpreter
            if self.use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if self.use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if self.use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (self.GRAPH_NAME == 'detect.tflite'):
                self.GRAPH_NAME = 'edgetpu.tflite'       

        # Get path to current working directory
        self.CWD_PATH = os.getcwd()

        # Path to .tflite file, which contains the model that is used for object detection
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH,self.MODEL_NAME,self.GRAPH_NAME)

        # Path to label map file        
        self.PATH_TO_LABELS = os.path.join(self.CWD_PATH,self.MODEL_NAME,self.LABELMAP_NAME)

        # Load the label map
        with open(self.PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if self.labels[0] == '???':
            del(self.labels[0])

        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        if self.use_TPU:
            self.interpreter = Interpreter(model_path=self.PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(self.PATH_TO_CKPT)
        else:
            self.interpreter = Interpreter(model_path=self.PATH_TO_CKPT)

        self.interpreter.allocate_tensors()

        ###########################################
        ########### Ariduino connection ###########
        ###########################################
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

        # Check output layer name to determine if this model was created with TF2 or TF1,
        # because outputs are ordered differently for TF2 and TF1 models
        self.outname = self.output_details[0]['name']

        if ('StatefulPartitionedCall' in self.outname): # This is a TF2 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 1, 3, 0
        else: # This is a TF1 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 0, 1, 2

        # Initialize frame rate calculation
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()

        # Initialize video stream
        self.videostream = VideoStream(resolution=(self.imW,self.imH),framerate=30).start()
        self.crosswalk_boundary = { 'x' : 520, 'xsize': 230, 'y' : 230, 'ysize' : 170}
        self.detect_box = { 'x' : 180, 'y' : 170, 'xsize' : 880, 'ysize' : 390}

        # flag for setting crosswalk boundary
        self.setCrosswalkBoundary()
        
        print('__ finished : initialization __')
        time.sleep(1)

    def run_detection(self) :
        #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        try:
            print('__ detection start __')
        # Start timer (for calculating frame rate)
            while True:
                t1 = cv2.getTickCount()
                # Grab frame from video stream
                frame1 = self.videostream.read()    
                # detect_box = { 'x' : 180, 'y' : 170, 'xsize' : 880, 'ysize' : 390}
                frame_for_disp = frame1.copy()
                
                boxes, classes, scores = self.detectWalkers(frame_for_disp.copy())
                
                # 횡단보도에 진입한 사람 숫자 반환.
                num_of_valid = self.checkEntering(boxes=boxes, scores=scores)
                ## 공통 파이프에 box 인풋.
            
                # draw rectangles
                self.drapeFrame(frame_for_disp=frame_for_disp, boxes=boxes, classes=classes, scores=scores)
                
                cv2.imshow('objectdetection', frame_for_disp)
                
                # Calculate framerate
                t2 = cv2.getTickCount()
                time1 = (t2-t1)/self.freq
                self.frame_rate_calc= 1/time1
                
                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q'):
                    break
        except Exception as e :
            print(e)
            return
        finally :
            # Clean up
            cv2.destroyAllWindows()
            self.videostream.stop()
        return

    def detectWalkers(self, mother_frame) :
        frame = mother_frame.copy()[self.detect_box['y']:self.detect_box['y'] + self.detect_box['ysize'], self.detect_box['x']:self.detect_box['x'] + self.detect_box['xsize']]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        # dectection_box = x, x, x, x
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std
    
        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
        self.interpreter.invoke()

        # Retrieve detection results
        boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])[0] # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])[0] # Confidence of detected objects
        return (boxes, classes, scores)


    def checkEntering(self, boxes, scores):
        global pipe
        enteringCount = 0
        for i in range(len(boxes)) :
            if not((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)): continue
            
            ymin = int(max(1,(boxes[i][0] * self.detect_box['ysize'])))
            xmin = int(max(1,(boxes[i][1] * self.detect_box['xsize'])))
            ymax = int(min(self.imH,(boxes[i][2] * self.detect_box['ysize'])))
            xmax = int(min(self.imW,(boxes[i][3] * self.detect_box['xsize'])))
            
            # calculate the coordinates of walker's foots
            walker_coords_on_absolute = [(xmin + xmax) / 2 + self.detect_box['x'], ymax + self.detect_box['y']]
            if (self.crosswalk_boundary['x'] < walker_coords_on_absolute[0] and self.crosswalk_boundary['x'] + self.crosswalk_boundary['xsize'] > walker_coords_on_absolute[0]) :
                if (self.crosswalk_boundary['y'] < walker_coords_on_absolute[1] and self.crosswalk_boundary['y'] + self.crosswalk_boundary['ysize'] / 2 > walker_coords_on_absolute[1]) :
                    # 하향 보행
                    pipe.append(1)
                    print(f'no.{i} walker is crossing the crosswalk in down direction.')
                    enteringCount += 1
                elif self.crosswalk_boundary['y'] + self.crosswalk_boundary['ysize'] / 2 < walker_coords_on_absolute[1] and self.crosswalk_boundary['y'] + self.crosswalk_boundary['ysize']  > walker_coords_on_absolute[1] :
                    # 상향 보행
                    pipe.append(2)
                    print(f'no.{i} walker is crossing the crosswalk in up direction.')
                    enteringCount += 1
        
        return enteringCount
            
            # 만약 아무도 횡단보도에 없다면 파이프에 데이터가 입력되지 않을 것.
    
    def drapeFrame(self, frame_for_disp, boxes, classes, scores) :
        # crosswalk_boundary = { 'x' : 520, 'xsize': 230, 'y' : 230, 'ysize' : 170}
        # detect_box = { 'x' : 80, 'y' : 170, 'xsize' : 1120, 'ysize' : 400}
    
        cv2.rectangle(frame_for_disp, (self.crosswalk_boundary['x'], self.crosswalk_boundary['y']), (self.crosswalk_boundary['x'] + self.crosswalk_boundary['xsize'], self.crosswalk_boundary['y'] + self.crosswalk_boundary['ysize']), (10, 255, 0), 2)
        cv2.rectangle(frame_for_disp, (self.detect_box['x'], self.detect_box['y']), (self.detect_box['x'] + self.detect_box['xsize'], self.detect_box['y'] + self.detect_box['ysize']), (255, 10, 10), 2)
        # Loop over all detections and draw detection box if confidence is above minimum threshold
    
        for i in range(len(scores)):
            if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * self.detect_box['ysize'])))
                xmin = int(max(1,(boxes[i][1] * self.detect_box['xsize'])))
                ymax = int(min(self.imH,(boxes[i][2] * self.detect_box['ysize'])))
                xmax = int(min(self.imW,(boxes[i][3] * self.detect_box['xsize'])))
    
                cv2.rectangle(frame_for_disp, (xmin + self.detect_box['x'],ymin + self.detect_box['y']), (xmax + self.detect_box['x'],ymax + self.detect_box['y']), (10, 255, 0), 2)

                # Draw label
                object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame_for_disp, (xmin + self.detect_box['x'], label_ymin-labelSize[1]-10 + self.detect_box['y']), (xmin+labelSize[0] + self.detect_box['x'], label_ymin+baseLine-10 + self.detect_box['y']), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame_for_disp, label + ' no . ' + str(i), (xmin + self.detect_box['x'] , label_ymin-7 + self.detect_box['y']), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                # Draw framerate in corner of frame
        cv2.putText(frame_for_disp,'FPS: {0:.2f}'.format(self.frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    
    def setCrosswalkBoundary(self) :
        cw_coords = []
        dt_coords = []
        frame = self.videostream.read()
        
        def mouseClick(event, x, y, flags, params) :
            global click_cnt
            if event == cv2.EVENT_FLAG_LBUTTON and click_cnt < 2 :
                print('mouse clicked', click_cnt)
                cv2.circle(frame, (x,y), 5, (255, 0, 0), 5)
                dt_coords.append((x,y))
                click_cnt += 1
                if click_cnt == 2 :
                    cv2.rectangle(frame,dt_coords[0], dt_coords[1], (255,0,0), 5)
            elif event == cv2.EVENT_FLAG_LBUTTON and click_cnt < 4:
                print('mouse clicked', click_cnt)
                cv2.circle(frame, (x,y), 5, (0,0,255), 5)
                cw_coords.append((x, y))
                click_cnt += 1
                if click_cnt == 4 :
                    cv2.rectangle(frame,cw_coords[0], cw_coords[1], (0,0,255), 5)
            cv2.imshow('crosswalk_setting', frame)
            
        cv2.namedWindow('crosswalk_setting')
        cv2.imshow('crosswalk_setting', frame)
        cv2.setMouseCallback('crosswalk_setting', mouseClick)
        try :
            while True:
                if cv2.waitKey(1) == ord('y') :
                    break
        except Exception as e:
            print(e)
        finally :
            cv2.destroyAllWindows()
        print(dt_coords, cw_coords)
        self.crosswalk_boundary['x'] = cw_coords[0][0]
        self.crosswalk_boundary['y'] = cw_coords[0][1]
        self.crosswalk_boundary['xsize'] = cw_coords[1][0] - cw_coords[0][0]
        self.crosswalk_boundary['ysize'] = cw_coords[1][1] - cw_coords[0][1]
        
        self.detect_box['x'] = dt_coords[0][0]
        self.detect_box['y'] = dt_coords[0][1]
        self.detect_box['xsize'] = dt_coords[1][0] - dt_coords[0][0]
        self.detect_box['ysize'] = dt_coords[1][1] - dt_coords[0][1]
        
        return
    

if __name__ == '__main__' :
    
    dc = DetectionController()
    objectDetectionThread = threading.Thread(target=dc.run_detection, args=())
    objectDetectionThread.start()
    ## arduino mediator