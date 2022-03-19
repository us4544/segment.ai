import streamlit as st

import torch
import torchvision
import torchvision.transforms as T

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import urllib
from PIL import Image, ImageDraw

from algorithm import instance_segmentation

# defining the instance segmentattion class
__COCO_INSTANCE_CATEGORY_NAMES__ = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

html_temp_2 = '''
    <div style = "padding-bottom: 20px; padding-top: 20px; padding-left: 20px; padding-right: 20px">      
    <center><h2>Instance Segmentation</h2></center>
    </div>
    '''
st.markdown(html_temp_2, unsafe_allow_html=True)
st.set_option('deprecation.showfileUploaderEncoding', False)
select = st.selectbox("Please select how you want to upload the image",("Please Select","Upload image via link","Upload image from device"))
if select == "Upload image via link":
    try:
        img = st.text_input('Enter the Image Address')
        img = Image.open(urllib.request.urlopen(img))
    except:
        if st.button('Submit'):
            show = st.error("Please Enter a valid Image Address!")
            time.sleep(4)
            show.empty()

if select == 'Upload image from device':
    file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file is not None:
        img = Image.open(file)

try:
    if img is not None:
        st.image(img, width = 300, caption = 'Uploaded Image')
        if st.button('Predict'):
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            ins_seg = instance_segmentation.InstanceSegmentation(model, img) #calling the instance segmentation class to run inference on the model
            ouptput_img = ins_seg.instance_segmentation()
            st.image(ouptput_img)
except:
    pass
