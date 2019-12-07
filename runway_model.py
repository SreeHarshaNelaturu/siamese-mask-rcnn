import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
sess_config = tf.ConfigProto()

import sys
import os

MASK_RCNN_MODEL_PATH = 'lib/Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)
    
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
    
from lib import utils as siamese_utils
from lib import model as siamese_model
from lib import config as siamese_config

import numpy as np
import skimage.io
import imgaug
import pickle
from collections import OrderedDict
import runway
from runway.data_types import *

class SmallEvalConfig(siamese_config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    NAME = 'random'
    EXPERIMENT = 'evaluation'
    NUM_TARGETS = 1
    
class LargeEvalConfig(siamese_config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    NAME = 'random'
    EXPERIMENT = 'evaluation'
    NUM_TARGETS = 1
    
    # Large image sizes
    TARGET_MAX_DIM = 192
    TARGET_MIN_DIM = 150
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Large model size
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    FPN_FEATUREMAPS = 256
    # Large number of rois at all stages
    RPN_ANCHOR_STRIDE = 1
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    TRAIN_ROIS_PER_IMAGE = 200
    DETECTION_MAX_INSTANCES = 100
    MAX_GT_INSTANCES = 100





# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
cat = category(choices=["small", "large"], default="small")

@runway.setup(options={"checkpoint" : file(extension=".h5"), "size" : cat})
def setup(opts):
    
    train_schedule = OrderedDict()
    model_size = opts["size"]
    
    if model_size == "small":
       config = SmallEvalConfig()
       config.display()
       train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
       train_schedule[120] = {"learning_rate": config.LEARNING_RATE, "layers": "4+"}
       train_schedule[160] = {"learning_rate": config.LEARNING_RATE/10, "layers": "all"}

    elif model_size == "large":
        config = LargeEvalConfig()
        config.display()
        train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
        train_schedule[240] = {"learning_rate": config.LEARNING_RATE, "layers": "all"}
        train_schedule[320] = {"learning_rate": config.LEARNING_RATE/10, "layers": "all"}
    
    print(train_schedule)
    checkpoint = opts["checkpoint"]

    config.NUM_TARGETS = 1

    model = siamese_model.SiameseMaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_checkpoint(checkpoint, training_schedule=train_schedule)

    return {"model" : model,
            "size" : model_size }

command_inputs = {"input_image" : image, "target_object" : image}
command_outputs = {"output_image" : image}


@runway.command("detect_target", inputs=command_inputs, outputs=command_outputs, description="One shot instance segmentation")
def detect_target(model, inputs):

    im = np.array(inputs["input_image"])
    target_im = np.array(inputs["target_object"].resize((96, 96) if model["size"] == "small" else (192, 192), Image.BICUBIC))
    print(target_im.shape)

    results = model["model"].detect([[target_im]], [im], verbose=1)
    r = results[0]

    out = siamese_utils.display_results(target_im, im, r['rois'], r['masks'], r['class_ids'], r['scores'])

    return {"output_image" : out}

if __name__ == "__main__":
    runway.run(model_options={"checkpoint" : "checkpoints/small_siamese_mrcnn_0160.h5", "size" : "small"})
