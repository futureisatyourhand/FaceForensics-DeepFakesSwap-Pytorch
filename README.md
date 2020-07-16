# FaceForensics-DeepFakes-Pytorch
pytorch-deepfakes/faceforensics/autoencoder
# plot.py
visiualizes the training processes of person1 and person2.

# models.py
declares the common encoder, the decoders of person1 and person2.
![Image text](https://github.com/futureisatyourhand/FaceForensics-DeepFakes-Pytorch/blob/master/deepfake.gif)
# training
We visulize our training process for two persons, the model trains 100000 epochs, and the picture shows the losses every 100 epochs.
![Image text](https://github.com/futureisatyourhand/FaceForensics-DeepFakes-Pytorch/blob/master/train.png)

# additional affine transformation
We need to construct the radiological transformation matrix according to the rotation center, rotation angle and scale.

We shift the center to the origin, and re-scale it, shift it.
Assuming that a sample rotate θ at the center point (x, y), and then enlarges or reduces by s, the radiative transformation matrix is:

![Image text](https://github.com/futureisatyourhand/FaceForensics-DeepFakes-Pytorch/blob/master/matrix.png)

The new face can be transformed into the target video based on the matrix to fool the human eyes.

