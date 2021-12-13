import argparse
import os
import sys

import numpy as np

from config import Config
from model import Model
from trimesh_renderer import TrimeshRenderer
from vis_util import preprocess_image, visualize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo HMR-SMPL')

    parser.add_argument('--image', required=False, default='hmr_test17.jpg')
    parser.add_argument('--model', required=False, default='base_model')
    parser.add_argument('--setting', required=False, default='ckpt')
    parser.add_argument('--joint_type', required=False, default='cocoplus')

    args = parser.parse_args()

    class DemoConfig(Config):
        BATCH_SIZE = 1
        ENCODER_ONLY = True
        CKPT_DIR = os.path.abspath('../models/{}/{}'.format(args.setting, args.model))
        JOINT_TYPE = args.joint_type

    config = DemoConfig()

    # initialize model
    model = Model()
    original_img, input_img, params = preprocess_image('../test_images/{}'.format(args.image), config.ENCODER_INPUT_SHAPE[0])

    result = model.detect(input_img)

    cam = np.squeeze(result['cam'].numpy())[:3]
    vertices = np.squeeze(result['vertices'].numpy())
    joints = np.squeeze(result['kp2d'].numpy())
    joints = ((joints + 1) * 0.5) * params['img_size']

    renderer = TrimeshRenderer()
    visualize(renderer, original_img, params, vertices, cam, joints)