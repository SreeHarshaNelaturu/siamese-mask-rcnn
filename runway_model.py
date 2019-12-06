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
from config_files import *

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

@runway.setup(options={"checkpoint" : file(extension=".h5"), "size" : category(choices=["small", "large"], default="small")})
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

    checkpoint = opts["checkpoint"]

    config.NUM_TARGETS = 1

    model = siamese_model.SiameseMaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_checkpoint(checkpoint, training_schedule=train_schedule)

    return {"model" : model,
            "size" : size }

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
    runway.run()
