# FaceForensics-DeepFakes-Pytorch
pytorch-deepfakes/faceforensics/autoencoder
# plot.py
visiualizes the training processes of person1 and person2.

# models.py
declares the common encoder, the decoders of person1 and person2.
![Image text](https://github.com/futureisatyourhand/FaceForensics-DeepFakes-Pytorch/blob/master/deepfake.gif)
# training
We visulize our training process for two persons, the model trains 10000 epochs, and the picture shows the losses every 100 epochs.
![Image text](https://github.com/futureisatyourhand/FaceForensics-DeepFakes-Pytorch/blob/master/train.png)
# Examples
Examples of training 9,000 epochs.

![Image text](https://github.com/futureisatyourhand/FaceForensics-DeepFakes-Pytorch/blob/master/9000.jpg)

# Pre-processing and post-processing
# pre-processing: remap and Umeyama additional affine transformation
remap: mapping the pixel of source to new images by some mechanism(such as gaussian distribution)

umeyama: Transformation of source cloud to target Cloud in the same coordinate system contains common matrix transformation and SVD decomposition process. After umeyama, we obtain the matrix for transformation. We input the original image, the matrix and the size into the warpAffine to get enhanced target image.

cv2.warpAffine: We need to construct the radiological transformation matrix according to the rotation center, rotation angle and scale.

We shift the center to the origin, and re-scale it, shift it.

Assuming that a sample rotate θ at the center point (x, y), and then enlarges or reduces by s, the radiative transformation matrix is:

![Image text](https://github.com/futureisatyourhand/FaceForensics-DeepFakes-Pytorch/blob/master/matrix.png)

The new face can be transformed into the target video based on the matrix to fool the human eyes.

# post-processing: we train the face part(64*64). After combing face of target A, decoded B and decoded A to forge the face A, the margin of forgered face is obvious.
Notes: If we train the head, the result will be fuzzy,it's difficult to restore.

method 1: smooth mask: gaussian smoothing

method 2: adjust avg color: A+=(mean(B)-mean(A))

method 3: Poisson distribution.


