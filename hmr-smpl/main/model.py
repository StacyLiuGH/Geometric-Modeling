import os
import sys
import time

# tf INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tqdm import tqdm

from config import Config
from generator import Generator
from model_util import batch_align_by_pelvis, batch_compute_similarity_transform, batch_rodrigues

class Model:

    def __init__(self):
        self.config = Config()
        self._build_model()

    def _build_model(self):

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        gen_input = ((self.config.BATCH_SIZE,) + self.config.ENCODER_INPUT_SHAPE)
		
        self.generator = Generator()
        self.generator.build(input_shape=gen_input)
        self.generator_opt = tf.optimizers.Adam(learning_rate=self.config.GENERATOR_LEARNING_RATE)

        self.checkpoint_prefix = os.path.join(self.config.CKPT_DIR, "ckpt")
        checkpoint = tf.train.Checkpoint(generator=self.generator,
                                             generator_opt=self.generator_opt)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, self.config.CKPT_DIR, max_to_keep=5)
       
        
        # if a checkpoint exists, restore the latest checkpoint.
        self.restore_check = None
        if self.checkpoint_manager.latest_checkpoint:
            restore_path = self.config.RESTORE_PATH
            if restore_path is None:
                restore_path = self.checkpoint_manager.latest_checkpoint

            self.restore_check = checkpoint.restore(restore_path).expect_partial()
            print('Checkpoint restored from {}'.format(restore_path))
        

    def detect(self, image):
        tf.keras.backend.set_learning_phase(0)

        if self.restore_check is None:
            raise RuntimeError('restore did not succeed, pleas check if you set config.CKPT_DIR correctly')

        if self.config.INITIALIZE_CUSTOM_REGRESSOR:
            self.restore_check.assert_nontrivial_match()
        else:
            self.restore_check.assert_existing_objects_matched().assert_nontrivial_match()

        if len(tf.shape(image)) != 4:
            image = tf.expand_dims(image, 0)

        result = self.generator(image, training=False)

        vertices_pred, kp2d_pred, kp3d_pred, pose_pred, shape_pred, cam_pred = result[-1]
        result_dict = {
            "vertices": tf.squeeze(vertices_pred),
            "kp2d": tf.squeeze(kp2d_pred),
            "kp3d": tf.squeeze(kp3d_pred),
            "pose": tf.squeeze(pose_pred),
            "shape": tf.squeeze(shape_pred),
            "cam": tf.squeeze(cam_pred)
        }
        return result_dict

if __name__ == '__main__':
    model = Model()
