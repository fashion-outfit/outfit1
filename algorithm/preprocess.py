import h5py
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from PIL import Image


def cropAndResizeToArray(inFile, x1, y1, x2, y2):
    with Image.open(inFile) as im:
        out = im.crop((x1, y1, x2, y2)).resize((128, 128))
        return np.array(out)

