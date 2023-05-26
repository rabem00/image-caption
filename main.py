import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

import cv2

# open mp4 file
vf = cv2.VideoCapture('tesla-bot-uprising.mp4')
#vf = cv2.VideoCapture('sample.mp4')
assert vf.isOpened(), 'The provided source cannot be captured.'

# set video frame size
vf.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vf.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Salesforce/blip-image-captioning-large
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# read video frame by frame
count = 0
while vf.isOpened():
    ret, frame = vf.read()
    if ret:
        cv2.imshow('video', frame)
        # Every 10 frames, caption the frame
        if count % 10 == 0:
            raw_image = Image.fromarray(frame).convert('RGB')          
            text = "a picture of"
            inputs = processor(raw_image, text, return_tensors="pt")
            out = model.generate(**inputs)
            print(processor.decode(out[0], skip_special_tokens=True))
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
