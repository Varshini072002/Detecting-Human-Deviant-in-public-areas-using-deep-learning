from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np
import os
import pywhatkit

app = Flask(__name__)

image_path = "last_frame_with_labels.jpg"
phone_number = "+918946002132"
# Route to upload video file and perform inference
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from twilio.rest import Client

account_sid = 'ACaf2dc38b125896263cc8c3c499a83a35'
auth_token = 'b6f643862c95b3d42973a3c06d60da78'
client = Client(account_sid, auth_token)

MODEL_CKPT = "archit11/videomae-base-finetuned-ucfcrime-full"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = VideoMAEForVideoClassification.from_pretrained(MODEL_CKPT).to(DEVICE)
PROCESSOR = VideoMAEFeatureExtractor.from_pretrained(MODEL_CKPT)

RESIZE_TO = PROCESSOR.size["shortest_edge"]
NUM_FRAMES_TO_SAMPLE = MODEL.config.num_frames
IMAGE_STATS = {"image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225]}
VAL_TRANSFORMS = Compose(
    [
        UniformTemporalSubsample(NUM_FRAMES_TO_SAMPLE),
        Lambda(lambda x: x / 255.0),
        Normalize(IMAGE_STATS["image_mean"], IMAGE_STATS["image_std"]),
        Resize((RESIZE_TO, RESIZE_TO)),
    ]
)
LABELS = list(MODEL.config.label2id.keys())


def parse_video(video_file):
    """A utility to parse the input videos.
    Reference: https://pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
    """
    vs = cv2.VideoCapture(video_file)

    # try to determine the total number of frames in the video file
    try:
        prop = (
            cv2.cv.CV_CAP_PROP_FRAME_COUNT
            if imutils.is_cv2()
            else cv2.CAP_PROP_FRAME_COUNT
        )
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    frames = []

    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

    return frames


def preprocess_video(frames: list):
    """Utility to apply preprocessing transformations to a video tensor."""
    # Each frame in the `frames` list has the shape: (height, width, num_channels).
    # Collated together the `frames` has the the shape: (num_frames, height, width, num_channels).
    # So, after converting the `frames` list to a torch tensor, we permute the shape
    # such that it becomes (num_channels, num_frames, height, width) to make
    # the shape compatible with the preprocessing transformations. After applying the
    # preprocessing chain, we permute the shape to (num_frames, num_channels, height, width)
    # to make it compatible with the model. Finally, we add a batch dimension so that our video
    # classification model can operate on it.
    video_tensor = torch.tensor(np.array(frames).astype(frames[0].dtype))
    video_tensor = video_tensor.permute(
        3, 0, 1, 2
    )  # (num_channels, num_frames, height, width)
    video_tensor_pp = VAL_TRANSFORMS(video_tensor)
    video_tensor_pp = video_tensor_pp.permute(
        1, 0, 2, 3
    )  # (num_frames, num_channels, height, width)
    video_tensor_pp = video_tensor_pp.unsqueeze(0)
    return video_tensor_pp.to(DEVICE)


def infer(video_file):
    frames = parse_video(video_file)
    video_tensor = preprocess_video(frames)
    inputs = {"pixel_values": video_tensor}

    # forward pass
    with torch.no_grad():
        outputs = MODEL(**inputs)
        logits = outputs.logits
    softmax_scores = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)
    confidences = {LABELS[i]: float(softmax_scores[i]) for i in range(len(LABELS))}
    return frames, confidences

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file.save(os.path.join('uploads', file.filename))
        saved_file_path = os.path.join('uploads', file.filename)
        #video_file = cv2.VideoCapture(saved_file_path)
        
        # Proceed with inference
        frames, result = infer(saved_file_path)

        sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)

        last_frame = frames[-1]

        for i in range(3):
            label, confidence = sorted_results[i]
            cv2.putText(last_frame, f"{label}: {confidence:.2f}", (20, 40 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
        output_file = "last_frame_with_labels.jpg"
        with open(output_file, 'wb') as f:
            f.write(buffer)
        message = client.messages.create(
        from_='whatsapp:+14155238886',
        body='Detected',
        to='whatsapp:+917539959347',
        media_url="https://c092-2409-40f4-3b-fdd6-7116-8787-10c7-40f2.ngrok-free.app/Final_app/New_anomaly/New_anomaly/last_frame_with_labels.jpg"
        )
        pywhatkit.sendwhats_image(phone_number,image_path,caption="detected")
        return send_file(output_file, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)

