import time
import sys
import warnings
import numpy
from skimage import io
import features
import color_space
import selective_search
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from os.path import join as pjoin
import os


def generate_color_table(R):
    # generate initial color
    colors = numpy.random.randint(0, 255, (len(R), 3))

    # merged-regions are colored same as larger parent
    for region, parent in R.items():
        if not len(parent) == 0:
            colors[region] = colors[parent[0]]

    return colors


data_root = '/home/ubelix/data/medical-labeling/'
data_dir = 'Dataset00'
data_frame_dir = 'input-frames'
im_name = 'frame_0400.png'
out_dir = 'results'


color = 'rgb'
feature = ['size', 'color', 'texture', 'fill']
output = 'result'
k = 50
alpha = 1.0

label = np.load(pjoin(data_root, data_dir, 'precomp_desc', 'sp_labels.npz'))['sp_labels'][..., 0]
image = io.imread(pjoin(data_root, data_dir, data_frame_dir, im_name))

print('k: {}'.format(k))
print('color: {}'.format(color))
print('feature: {}'.format(feature))

start_t = time.time()
mask = features.SimilarityMask('size' in feature, 'color' in feature,
                               'texture' in feature, 'fill' in feature)
#R: stores region label and its parent (empty if initial).
# record merged region (larger region should come first)
(R, F, g) = selective_search.hierarchical_segmentation(
    image,
    k=k,
    feature_mask=mask,
    F0=label,
    return_stacks=True)

# suppress warning when saving result images
warnings.filterwarnings("ignore", category=UserWarning)

end_t = time.time()
print('Built hierarchy in ' + str(end_t - start_t) + ' secs')

print('result filename: %s_[0000-%04d].png' % (output, len(F) - 1))

if(os.path.exists(out_dir)):
   print('Deleting content of dir: ' + out_dir)
   fileList = os.listdir(out_dir)
   for fileName in fileList:
       #os.remove(out_dir+"/"+fileName)
       os.remove(os.path.join(out_dir,fileName))

if(not os.path.exists(out_dir)):
   print('output directory does not exist... creating')
   os.mkdir(out_dir)

print('Saving images to dir: ' + str(out_dir))
colors = generate_color_table(R)
pbar = tqdm.tqdm(total=len(F))
for depth, label in enumerate(F):
   result = colors[label]
   result = (result * alpha + image * (1. - alpha)).astype(numpy.uint8)
   fn = "%s_%04d.png" % (os.path.splitext(im_name)[0], depth)
   fn = os.path.join(out_dir,fn)
   io.imsave(fn, result)
   pbar.update(1)
