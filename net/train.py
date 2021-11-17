import os
import argparse
from typing import Tuple
from datetime import datetime

import tensorflow as tf
import segmentation_models as sm

from tensorflow.keras.callbacks import EarlyStopping

from config import (JSON_FILE_NAME, EPOCHS, LEARNING_RATE, SAVE_MODELS, INPUT_SHAPE_IMAGE, MODEL_NAME,
                    BACKBONE, LOGS)
from src import build_model, DataGenerator, LogCallback
sm.set_framework('tf.keras')
sm.framework()

def train(data_path: str, input_shape_image: Tuple[int, int, int] = INPUT_SHAPE_IMAGE) -> None:
    """
    Training to classify generated images.

    :param data_path: data path.
    :param input_shape_image: this is images shape (height, width, channels).
    """
    date_time_for_save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.models_data, '{}_{}_{}_shape-{}'.format(MODEL_NAME, BACKBONE, date_time_for_save,
                                                                          str(input_shape_image[0]) + '-' +
                                                                          str(input_shape_image[1])))
    save_current_logs = os.path.join(save_path, LOGS)
    save_current_model = os.path.join(save_path, SAVE_MODELS)

    # create dirs
    for p in [save_path, save_current_model, save_current_logs]:
        os.makedirs(p, exist_ok=True)

    train_data_gen = DataGenerator(data_path=data_path, json_name=JSON_FILE_NAME, is_train=True,
                                   image_shape=input_shape_image)
    test_data_gen = DataGenerator(data_path=data_path, json_name=JSON_FILE_NAME, is_train=False,
                                  image_shape=input_shape_image)

    model = build_model(image_shape=input_shape_image)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss=sm.losses.categorical_focal_jaccard_loss,
                  metrics=['accuracy', sm.metrics.iou_score, sm.metrics.precision, sm.metrics.recall,
                           sm.metrics.f1_score]
                  )
    model.summary()
    early = EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=1, mode='auto')
    checkpoint_filepath = os.path.join(save_current_model, MODEL_NAME + BACKBONE + '.h5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_iou_score',
        mode='max',
        save_best_only=True
    )

    tensor_board = tf.keras.callbacks.TensorBoard(save_current_logs, update_freq='batch')
    with LogCallback(logs_save_path=save_current_logs, model_save_path=save_current_model,
                     input_shape=input_shape_image) as call_back:
        model.fit_generator(generator=train_data_gen, validation_data=test_data_gen, validation_freq=1,
                            validation_steps=len(test_data_gen), epochs=EPOCHS, workers=8,
                            callbacks=[early, model_checkpoint_callback, tensor_board, call_back])


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('script for model testing.')
    parser.add_argument('--train', default=True, action='store_true',
                        help='if you use this flag, then the training will be performed automatically with input shape'
                             '(256*256*3))')
    parser.add_argument('--train_dif_shape', default=False, action='store_true',
                        help='if you use this flag, then the training will be performed automatically with a different'
                             'input shape from (256*256*3) before (512*512*3)')
    parser.add_argument('--data_path', type=str, default='data/first_data', help='path to Dataset where there is a json file')
    parser.add_argument('--models_data', type=str, default='models_data', help='path for saving logs and models')
    parser.add_argument('--gpu', type=str, default='0', help='If you want to use the GPU, you must specify the number '
                                                             'of the video card that you want to use.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    #
    # if args.gpu != '_':
    #     gpus_list = [int(args.gpu)]
    # else:
    #     gpus_list = []
    # devices = tf.config.get_visible_devices('GPU')
    # devices = [devices[i] for i in gpus_list]
    # tf.config.set_visible_devices(devices, 'GPU')
    # for gpu in devices:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    if args.train is True:
        train(data_path=args.data_path, input_shape_image=INPUT_SHAPE_IMAGE)

    if parse_args().train_dif_shape is True:
        for i in range(5):
            input_shape = [(INPUT_SHAPE_IMAGE[0] + (64 * i), INPUT_SHAPE_IMAGE[1] +
                            (64 * i), 3) for i in range(0, 5, 1)]
            train(data_path=args.data_path, input_shape_image=input_shape[i])
