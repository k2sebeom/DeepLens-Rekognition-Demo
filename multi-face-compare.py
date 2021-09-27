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
from local_display import LocalDisplay

import asyncio
import concurrent.futures


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


def mark_face(image, bbox, label=""):
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

async def get_inference(img_bytes, loop, executor, client, face_keys):
    blocking_tasks = []
    for img_key in face_keys:
        blocking_tasks.append(loop.run_in_executor(executor, lambda: get_resp(client, img_key, img_bytes)))
    completed, pending = await asyncio.wait(blocking_tasks)
    results = [t.result() for t in completed]
    return results

def get_resp(client, img_key, img_bytes):
    try:
        return {
            "key": img_key,
            "resp": client.compare_faces(
                        SourceImage={
                            'S3Object': {
                                'Bucket': '{s3 bucket name}',
                                'Name': img_key
                            }
                        },
                        TargetImage={
                            'Bytes': img_bytes
                        }
            )
        }
    except Exception:
        return None

def infinite_infer_run():
    """ Entry point of the lambda function"""
    try:
        # This face detection model is implemented as single shot detector (ssd).
        model_type = 'ssd'
        output_map = {1: 'face'}
        # Create an IoT client for sending to messages to the cloud.
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

        event_loop = asyncio.get_event_loop()

        target_dict = {
            "list.jpeg": "Names",
            "of.png": "of",
            "s3_keys.jpeg": "faces"
        }

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
            parsed_inference_results = model.parseResult(model_type,
                                                         model.doInference(frame_resize))
            # Dictionary to be filled with labels and probabilities for MQTT
            cloud_output = {}
            # Get the detected faces and probabilities
            for obj in parsed_inference_results[model_type]:
                if obj['prob'] > detection_threshold:
                    # Store label and probability to send to cloud
                    cloud_output[output_map[obj['label']]] = obj['prob']
            if cloud_output:
                
                ret, jpeg = cv2.imencode(".jpg", frame)
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    results = event_loop.run_until_complete(get_inference(jpeg.tobytes(), event_loop, executor, rekog, target_dict.keys()))
                others_logged = False
                for i, result in enumerate(results):
                    if not result:
                        continue
                    key, resp = result["key"], result["resp"]
                    matches = resp["FaceMatches"]
                    
                    for face in matches:
                        bbox = face["Face"]["BoundingBox"]
                        mark_face(frame, bbox, label=target_dict[key])

                    if not others_logged:
                        unmatches = resp["UnmatchedFaces"]
                        for face in unmatches:
                            bbox = face["BoundingBox"]
                            mark_face(frame, bbox, label="")
                        others_logged = True

            
            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
            # Send results to the cloud
            client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
    except Exception as ex:
        client.publish(topic=iot_topic, payload='Error in face detection lambda: {}'.format(ex))

infinite_infer_run()