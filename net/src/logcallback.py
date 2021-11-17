import os
import json
from typing import Dict, Tuple

from tensorflow import keras

from config import (SAVE_MODEL_EVERY_EPOCH, BATCH_SIZE, LEARNING_RATE, BACKBONE, ENCODER_WEIGHTS, OUTPUT_ACTIVATION,
                    MODEL_NAME, AUGMENTATION_DATA, LAYER_ACTIVATION)


class LogCallback(keras.callbacks.Callback):
    def __init__(self, model_save_path: str, logs_save_path: str, input_shape: Tuple, model_name: str = MODEL_NAME,
                 batch_size: int = BATCH_SIZE, backbone: str = BACKBONE, learning_rate: float = LEARNING_RATE,
                 encoder_weights: str = ENCODER_WEIGHTS, save_model_every_era: bool = SAVE_MODEL_EVERY_EPOCH,
                 augmentation_data: bool = AUGMENTATION_DATA, output_activation: str = OUTPUT_ACTIVATION,
                 layer_activation: str = LAYER_ACTIVATION) -> None:
        """
        Logging all training metrics to a json file and saving the model every epoch.

        :param model_save_path: path to the folder in which to save the experemental_model.
        :param logs_save_path: path to the folder in which to save the logs.
        :param save_model_every_era: save the model or not at the end of each epoch.
        :param input_shape: input shape images.
        :param model_name: the name of the model that is used as the basis (Unet ...).
        :param batch_size: number of images in one batch.
        :param backbone: name of classification model (without last dense layers) used as feature extractor to
                          build segmentation model.
        :param learning_rate: learning rate for optimizer.
        :param encoder_weights: whether imagenet weights are used or not.
        :param augmentation_data: whether augmentation is used or not.
        :param output_activation: this is output activation.
        :param layer_activation: this is layer activation.

        """
        super().__init__()
        self.model_save_path = model_save_path
        self.log_file = os.path.join(logs_save_path, 'train_logs.json')
        self.save_model_every_epoch = save_model_every_era
        self.logs = {'epochs': [],
                     'Model_name': model_name,
                     'Backbone': backbone,
                     'Input_shape': input_shape,
                     'Batch_size': batch_size,
                     'Learning_rate': learning_rate,
                     'Encoder_weights': encoder_weights,
                     'Augmentation_data': augmentation_data,
                     'Output_activation': output_activation,
                     'Layer_activation': layer_activation
                     }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_logs()
        if self.save_model_every_epoch:
            self.model.save(os.path.join(self.model_save_path, 'last.h5'))
        # self.neural_network_parameters()

    def save_logs(self) -> None:
        with open(self.log_file, 'w') as file:
            json.dump(self.logs, file, indent=4)

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None) -> None:
        """
        This method prepares the data for writing to a json file and calls the method to save the json file.
        If  = self.save_model_every_epoch = True, then the model is saved every epoch.

        :param epoch: current epoch number.
        :param logs: dictionary with metrics that are specified in the model, such as accuracy, precision or recall.
        """
        text = ['epoch: {:03d}'.format(epoch + 1)]
        for key, value in logs.items():
            if key not in self.logs:
                self.logs[key] = []
            self.logs[key].append(float(value))
            text.append('{}: {:.04f}'.format(key, float(value)))
        self.logs['epochs'].append('; '.join(text))
        self.save_logs()
        if self.save_model_every_epoch:
            self.model.save(os.path.join(self.model_save_path, '{:03d}_epoch.h5'.format(1 + epoch)))
