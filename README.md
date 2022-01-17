# Face Mask Detection with MobileNetV2 
## _Statistic under AI and its Application to Engineering Sciences_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://rauzansumara.github.io/)

This report is trying to create a robust face mask detection based on convolution neural network. In order to detect a face mask, we implement the object detection algorithm which is MobileNetV2 architecture. The rest of this report will be organized as follows: section two illustrates the datasets. Section three describes the face mask detection algorithm and section four presents the experimental results in real-time data. This work will be closed by the conclusion in section five

### Requirements
```
python==3.8.5
tensorflow==2.4.0
keras==2.4.3
numpy==1.21.4
pandas==1.3.5
sklearn==0.24.2
matplotlib==3.3.2
tqdm==4.41.1
```

## Datasets
The dataset used for this task was created by  [Prajna Bhandary](https://github.com/prajnasb/observations). The dataset contains 1,376 images belonging to two classes, with mask: 690 images and without mask: 686 images. images without mask are actually artificial face mask datasets based on applying facial landmarks. Facial landmarks allow us to automatically infer the location of facial structures, including, eyes, eyebrows, nose, mouth, jawline, etc. The size of the training set chosen for this task is 80% of the images and the test set consists of remaining 20% of the images

![](https://raw.githubusercontent.com/rauzansumara/face-mask-detection-based-on-mobilenetv2/master/images/print_images.png)

## The MobileNetV2 Architecture
As for the face mask detector method, this work implemented the convolutional neural network known as MobileNetV2 architecture. According to Sandler, et al. (2018), MobileNetV2 is a convolutional neural network architecture that seeks to perform well on mobile devices. It is based on an inverted residual structure where the residual connections are between the bottleneck layers. The intermediate expansion layer uses lightweight depth wise convolutions to filter features as a source of non-linearity. As a whole, the architecture of MobileNetV2 contains the initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers. By reflecting these results, it is suitable to implement the method into the real-time face mask detector where high detection accuracy is needed.

## Experimental Results
We conducted our experiment using python programming language. You can see that the validation loss and validation accuracy both are in sync with the training loss and training accuracy. The training and validation loss are decreasing and there is not much gap between training and validation accuracy. It shows that our model is not overfitting:

![](https://raw.githubusercontent.com/rauzansumara/face-mask-detection-based-on-mobilenetv2/master/images/training.png)

- Import a HTML file and watch it magically convert to Markdown
- Drag and drop images (requires your Dropbox account be linked)
- Import and save files from GitHub, Dropbox, Google Drive and One Drive
- Drag and drop markdown and HTML files into Dillinger
- Export documents as Markdown, HTML and PDF

## Implement Model on Selected instances of Images
we provided two examples of images to evaluate the model. After running the model following results were obtained:

![](https://raw.githubusercontent.com/rauzansumara/face-mask-detection-based-on-mobilenetv2/master/images/example1_mask.png)

![](https://raw.githubusercontent.com/rauzansumara/face-mask-detection-based-on-mobilenetv2/master/images/example2_mask.png) 

## Implement Model on Real-Time Video
![](https://raw.githubusercontent.com/rauzansumara/face-mask-detection-based-on-mobilenetv2/master/images/mask.png) 

![](https://raw.githubusercontent.com/rauzansumara/face-mask-detection-based-on-mobilenetv2/master/images/no_mask.png) 

## Implement Model on Based-Android Application
In this case of implementing our model on based-android app, we need to implement the two steps detection. Most of the work will consist in splitting the detection, first the face detection and second the mask detection. For the face detection step, we are going to use the Google ML kit. For the mask detection, we are going to use our model. First thing to do is using TocoConverter python class to migrate from the Keras .h5 format model to the TensorFlow Lite .tflite format model. It’s amazing how easy a high-level deep learning model can be ported to format suitable for mobile, simply by executing one line of code. The model was created in previous section, producing a ‘.h5’ file of about 11.2 MB. After TensorFlow Lite conversion, the resulting file is very light-weight only 9.2 MB, really good for a mobile application. Here is the example of the working app on my Phone (Android Version 6.0.1).

![](https://raw.githubusercontent.com/rauzansumara/face-mask-detection-based-on-mobilenetv2/master/images/demo.gif)  


