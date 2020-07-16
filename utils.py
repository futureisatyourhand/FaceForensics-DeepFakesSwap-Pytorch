import numpy as np
import cv2
##get images dataset
def load_images(paths,convert=None):
    all_images=[cv2.resize(cv2.imread(f),(256,256)) for f in paths]
    if convert:
        all_images=[convert(img) for img in all_images]
    return np.array(all_images)

def get_transpose_axes(n):
    if n%2==0:
        y=list(range(1,n-1,2))
        x=list(range(0,n-1,2))
    else:
        x=list(range(1,n-1,2))
        y=list(range(0,n-1,2))
    return y,x,[n-1]

def stack_images(images):
    images_shape=np.array(images.shape)
    new_axes=get_transpose_axes(len(images_shape))
    print(images_shape,new_axes)
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    return np.transpose(
        images,
        axes=np.concatenate(new_axes)
    ).reshape(new_shape)
