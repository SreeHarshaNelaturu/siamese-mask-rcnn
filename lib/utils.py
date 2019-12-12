# Simaese Mask R-CNN Utils

import tensorflow as tf
import sys
import os
import time
import random
import numpy as np
import skimage.io
import skimage.transform as skt
import imgaug
import matplotlib.pyplot as plt
import tempfile
from PIL import Image
plt.rcParams['figure.figsize'] = (12.0, 6.0)

MASK_RCNN_MODEL_PATH = 'Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)
    

from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize  

import warnings
warnings.filterwarnings("ignore")
    

def display_results(target, image, boxes, masks, class_ids,
                      scores=None, title="",
                      figsize=(8, 8), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        auto_show = True

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
   
    masked_image = image.astype(np.uint32).copy()
    out_masks = []
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{:.3f}".format(score) if score else 'no score'
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = visualize.find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = visualize.Polygon(verts, facecolor="none", edgecolor=color)
            #print(p)
            ax.add_patch(p)
        out_masks.append(padded_mask)
   
    ax.imshow(masked_image.astype(np.uint8))
    ax.axis('tight')
    ax.axis('off')

    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.margins(0,0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    mask_imgs = []
    for i in out_masks:
        #a = np.expand_dims(i*255, -1)
        stacked_img = np.stack((i*255,)*3, axis=-1).astype(np.uint8)
        mask_img = Image.fromarray(stacked_img) 
        mask_imgs.append(mask_img)

    return image_from_plot, mask_imgs



