from threading import Thread, Event
import os
import json
import numpy as np
import awscam
import cv2
import greengrasssdk
import sys
import urllib
import zipfile
import time


# Latest software has boto3 installed
try:
    import boto3
except Exception:
    boto_dir = '/tmp/boto_dir'
    if not os.path.exists(boto_dir):
        os.mkdir(boto_dir)
    urllib.urlretrieve("https://s3.amazonaws.com/dear-demo/boto_3_dist.zip", "/tmp/boto_3_dist.zip")
    with zipfile.ZipFile("/tmp/boto_3_dist.zip", "r") as zip_ref:
        zip_ref.extractall(boto_dir)
    sys.path.append(boto_dir)
    import boto3


def mark_object(image, bbox, label=""):
    H, W, _ = image.shape
    x = int(bbox["Left"] * W)
    y = int(bbox["Top"] * H)
    w = int(bbox["Width"] * W)
    h = int(bbox["Height"] * H)
    
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 165, 20), 10)
    text_offset = 15
    cv2.putText(image, label,
                (x, y - text_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 165, 20), 3)    

def infinite_infer_run():
    """ Entry point of the lambda function"""
    try:
        client = greengrasssdk.client('iot-data')
        iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()
        # The sample projects come with optimized artifacts, hence only the artifact
        # path is required.
        model_path = '/opt/awscam/artifacts/mxnet_deploy_ssd_FP16_FUSED.xml'
        # Load the model onto the GPU.
        client.publish(topic=iot_topic, payload='Loading face detection model')
        model = awscam.Model(model_path, {'GPU': 1})
        client.publish(topic=iot_topic, payload='Face detection model loaded')
        
        rekog = boto3.client("rekognition")
        client.publish(topic=iot_topic, payload='Rekognition client crearted')
        # Set the threshold for detection
        detection_threshold = 0.25
        # The height and width of the training set images
        input_height = 300
        input_width = 300
        # Do inference until the lambda is killed.
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
            # Resize frame to the same size as the training set.
            frame_resize = cv2.resize(frame, (input_height, input_width))
            # Run the images through the inference engine and parse the results using
            # the parser API, note it is possible to get the output of doInference
            # and do the parsing manually, but since it is a ssd model,
            # a simple API is provided.
            ret, jpeg = cv2.imencode(".jpg", frame_resize)
            resp = rekog.detect_labels(
                Image={
                    'Bytes': jpeg.tobytes()
                },
                MaxLabels=10
            )
            # print(resp)
            for obj in resp['Labels']:
                for inst in obj['Instances']:
                    bbox = inst['BoundingBox']
                    label = obj["Name"]
                    conf = obj['Confidence']
                    mark_object(frame_resize, bbox, label=f"{label} {conf:.2f}%")

            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
            # Send results to the cloud
            client.publish(topic=iot_topic, payload=json.dumps(resp))
    except Exception as ex:
        client.publish(topic=iot_topic, payload='Error in face detection lambda: {}'.format(ex))

infinite_infer_run()

