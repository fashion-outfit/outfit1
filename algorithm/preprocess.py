import json

import h5py
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from PIL import Image


def cropAndResizeToArray(inFile, x1, y1, x2, y2):
    with Image.open(inFile) as im:
        out = im.crop((x1, y1, x2, y2)).resize((128, 128))
        return np.array(out)


def getAttrLoc():
    with open('Anno/list_attr_cloth.txt') as f:
        f.readline()
        f.readline()
        attrCloth = {
            'texture': {},
            'fabric': {},
            'shape': {},
            'part': {},
            'style': {}
        }
        for i, line in enumerate(f.readlines()):
            attr = line.split('  ')[0]
            if '1' in line:
                attrCloth['texture'][i] = attr
            if '2' in line:
                attrCloth['fabric'][i] = attr
            if '3' in line:
                attrCloth['shape'][i] = attr
            if '4' in line:
                attrCloth['part'][i] = attr
            if '5' in line:
                attrCloth['style'][i] = attr
        with open('Anno/list_attr_cloth.json', 'w') as f:
            json.dump(attrCloth, f)


def getImgFeature():
    with open('Anno/list_attr_cloth.json') as f:
        attrCloth = json.load(f)

    # dump style attrs
    with open('Anno/list_attr_img.txt') as f:
        f.readline()
        f.readline()
        style = {}
        for line in f.readlines():
            lineSplitted = line.strip().split(' ')
            while '' in lineSplitted:
                lineSplitted.remove('')
            imgPath = lineSplitted.pop(0)
            style[imgPath] = []
            for i in attrCloth['style']:
                style[imgPath].append(lineSplitted[i])

