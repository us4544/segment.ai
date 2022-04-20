# Instance Segmentation using Mask RCNN
Image segmentation creates a pixel-wise mask for each object in the image. This technique gives us a far more granular understanding of the object(s) in the image. This can be done through various techniques like:
- Threshold Based Segmentation
- Edge Based Segmentation
- Region-Based Segmentation
- Clustering Based Segmentation
- Artificial Neural Network Based Segmentation

![image](https://user-images.githubusercontent.com/66861243/159152309-61ec369c-aae8-4f20-b25e-a2329745248e.png)

## Mask RCNN
Mask R-CNN extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework.

### Network Architecture
- A convolutional backbone architecture used for feature extraction over an entire image
- A network head for bounding-box recognition (classification and regression) and mask prediction that is applied separately to each RoI.

![image](https://user-images.githubusercontent.com/66861243/159152466-d6e859d7-198a-423b-8d0c-8f99c8732748.png)

### Results from original Mask RCNN Implementation by Facebook AI Research
![image](https://user-images.githubusercontent.com/66861243/159152504-857e8e0c-362f-43aa-aefd-6d028cfb0865.png)

## Demo
<img src = "results/results.png">

## Installation and Quick Start

- Setting up the Python Environment with dependencies:

        pip install -r requirements.txt

- Cloning the Repository: 

        git clone https://github.com/us4544/Segment.AI
- Entering the directory: 

        cd Segment.AI
- Running the file:

        pip install virtualenv
        python -m venv env
        source env/bin/activate
        pip install -r requirements.txt
        streamlit run main.py
        
<hr>
