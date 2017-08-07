#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import sys
import warnings
import numpy
import skimage.io
import features
import color_space
import selective_search
from scipy import (ndimage,io)
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import my_utils as utls



def generate_color_table(R):
    # generate initial color
    colors = numpy.random.randint(0, 255, (len(R), 3))

    # merged-regions are colored same as larger parent
    for region, parent in R.items():
        if not len(parent) == 0:
            colors[region] = colors[parent[0]]

    return colors


dataInRoot = '/home/laurent.lejeune/medical-labeling/'
dataSetDir = 'Dataset2'
frameDir = 'input-frames'
im_name = 'frame_0140.png'
out_dir = 'results'

image = os.path.join(dataInRoot,dataSetDir,frameDir,im_name)

color = 'rgb'
feature = ['desc']
output = 'result'
k = 50
alpha = 1.0
img = skimage.io.imread(image)

label = io.loadmat(os.path.join(dataInRoot,                                      dataSetDir,                                          frameDir,'sp_labels.mat'))['sp_labels'][...,0]


if len(img.shape) == 2:
    img = skimage.color.gray2rgb(img)

print('k:', k)
print('color:', color)
print('feature:', ' '.join(feature))

dir_res = '/home/laurent.lejeune/medical-labeling/Dataset2/results/2017-07-21_11-39-17_exp'

with open(os.path.join(dir_res, 'cfg.yml'), 'r') as outfile:
    conf = yaml.load(outfile)

import pdb; pdb.set_trace()
my_dataset = ds.Dataset(conf)

start_t = time.time()
mask = features.SimilarityMask('size' in feature, 'color' in feature, 'texture' in feature, 'fill' in feature, 'desc' in feature)
#R: stores region label and its parent (empty if initial).
# record merged region (larger region should come first)
sp_desc_df = my_dataset.sp_desc_df
(R, F, L, A, S) = selective_search.hierarchical_segmentation(img, k, mask,F0=label, desc_arr=sp_desc_df[sp_desc_df['frame'] == f]['desc'].as_matrix())

# suppress warning when saving result images
warnings.filterwarnings("ignore", category = UserWarning)

end_t = time.time()
print('Built hierarchy in ' + str(end_t - start_t) + ' secs')

print('result filename: %s_[0000-%04d].png' % (output, len(F) - 1))

S = [(0,483),(0,313)]

import pdb; pdb.set_trace()
import my_utils as utls
utls.make_sc_graph(R)


#if(os.path.exists(out_dir)):
#    print('Deleting content of dir: ' + out_dir)
#    fileList = os.listdir(out_dir)
#    for fileName in fileList:
#        #os.remove(out_dir+"/"+fileName)
#        os.remove(os.path.join(out_dir,fileName))
#
#if(not os.path.exists(out_dir)):
#    print('output directory does not exist... creating')
#    os.mkdir(out_dir)
#
#print('Saving images to dir: ' + str(out_dir))
#start_t = time.time()
#colors = generate_color_table(R)
#for depth, label in enumerate(F):
#    result = colors[label]
#    result = (result * alpha + img * (1. - alpha)).astype(numpy.uint8)
#    fn = "%s_%04d.png" % (os.path.splitext(im_name)[0], depth)
#    fn = os.path.join(out_dir,fn)
#    skimage.io.imsave(fn, result)
#    print('.', end="")
#    sys.stdout.flush()
#end_t = time.time()
#print()
#print('Saved images in ' + str(end_t - start_t) + ' secs')
