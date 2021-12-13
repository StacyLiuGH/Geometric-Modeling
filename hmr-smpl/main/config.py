import json
from datetime import datetime

import os

class Config(object):
    __instance = None

    def __new__(cls):
        if Config.__instance is None:
            Config.__instance = object.__new__(cls)
            Config.__instance.__initialized = False

        return Config.__instance

    # ------Directory settings:------
    #
    # root project directory
    ROOT_PROJECT_DIR = os.path.abspath(os.path.join(__file__, '..', '..'))

    RESTORE_PATH = None

    # path to the neutral smpl model
    # ['neutral_smpl_coco_regressor.pkl']
    SMPL_MODEL_PATH = os.path.join(ROOT_PROJECT_DIR, 'models', 'neutral_smpl_coco_regressor.pkl')

    # path to mean theta h5 file
    SMPL_MEAN_THETA_PATH = os.path.join(ROOT_PROJECT_DIR, 'models', 'neutral_smpl_mean_params.h5')

    # path to the custom regressors
    CUSTOM_REGRESSOR_PATH = os.path.join(ROOT_PROJECT_DIR, 'models', 'regressors')

    # ------HMR parameters:------
    #
    # input image size to the encoder network after preprocess
    ENCODER_INPUT_SHAPE = (224, 224, 3)

    # number of iterations for regressor feedback loop
    ITERATIONS = 3

    # define joint type returned by SMPL
    # any of [cocoplus, lsp, custom, coco_custom]
    JOINT_TYPE = 'cocoplus'

    # cocoplus: 19 keypoints
    # lsp:  14 keypoints
    # custom: set keypoints according to generated regressors
    DS_KP2D = {
        'lsp': 14,
        'cocoplus': 19
    }
    DS_KP3D = {
        'lsp': 14,
        'cocoplus': 14
    }

    # if you want to run inference or evaluation with a pretrained standard lsp or cocoplus model
    # but still regress for the new keypoints set this to True
    INITIALIZE_CUSTOM_REGRESSOR = False

    # if set to True, no adversarial prior is trained = monsters
    ENCODER_ONLY = False

    # ------Hyper parameters:------
    #
    # generator learning rate
    GENERATOR_LEARNING_RATE = 1e-5
  
    GENERATOR_WEIGHT_DECAY = 1e-4 

    # weight on generator 2d loss
    GENERATOR_2D_LOSS_WEIGHT = 60.
    # weight on generator 3d loss
    GENERATOR_3D_LOSS_WEIGHT = 60.

    '''
    # adversarial prior learning rate
    DISCRIMINATOR_LEARNING_RATE = 1e-4
    # adversarial prior weight decay
    DISCRIMINATOR_WEIGHT_DECAY = 1e-4
    # weight on discriminator
    DISCRIMINATOR_LOSS_WEIGHT = 1
    '''

    # ------SMPL settings:------
    #
    # number of smpl joints
    NUM_JOINTS = 23
    NUM_JOINTS_GLOBAL = NUM_JOINTS + 1

    # number of cameras parameters [scale, tx, ty]
    NUM_CAMERA_PARAMS = 3

    # The pose (theta) is modeled by relative 3D rotation
    # of K joints in axis-angle representation (rotation matrix)
    # K joints + 1 (global rotation)
    # override this according to smpl representation
    NUM_POSE_PARAMS = NUM_JOINTS_GLOBAL * 3

    # number of shape (beta) parameters
    # override this according to smpl representation
    NUM_SHAPE_PARAMS = 10

    # total number of smpl params
    NUM_SMPL_PARAMS = NUM_CAMERA_PARAMS + NUM_POSE_PARAMS + NUM_SHAPE_PARAMS

    # total number of vertices
    NUM_VERTICES = 6890

    def __init__(self):
        if self.__initialized:
            return

        self.__initialized = True

        self.NUM_KP2D = self.DS_KP2D.get(self.JOINT_TYPE)
        self.NUM_KP3D = self.DS_KP3D.get(self.JOINT_TYPE)

    def reset(self):
        Config.__instance = None
