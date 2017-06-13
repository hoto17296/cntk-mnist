# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import sys
import os
import cntk as C

# Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(abs_path, "data")
model_path = os.path.join(abs_path, "models")

# Define the reader for both training and evaluation action.
def create_reader(path, is_training, input_dim, label_dim):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        features=C.io.StreamDef(field='features', shape=input_dim),
        labels=C.io.StreamDef(field='labels',   shape=label_dim)
    )), randomize=is_training, max_sweeps=C.io.INFINITELY_REPEAT if is_training else 1)


# Creates and trains a feedforward classification model for MNIST images
def convnet_mnist(debug_output=False, epoch_size=60000, minibatch_size=64, max_epochs=40, block_size=3200):
    image_height = 28
    image_width  = 28
    num_channels = 1
    input_dim = image_height * image_width * num_channels
    num_output_classes = 10

    # Input variables denoting the features and label data
    input_var = C.ops.input_variable((num_channels, image_height, image_width), np.float32)
    label_var = C.ops.input_variable(num_output_classes, np.float32)

    # Instantiate the feedforward classification model
    scaled_input = C.ops.element_times(C.ops.constant(0.00390625), input_var)

    with C.layers.default_options(activation=C.ops.relu, pad=False):
        conv1 = C.layers.Convolution2D((5,5), 32, pad=True)(scaled_input)
        pool1 = C.layers.MaxPooling((3,3), (2,2))(conv1)
        conv2 = C.layers.Convolution2D((3,3), 48)(pool1)
        pool2 = C.layers.MaxPooling((3,3), (2,2))(conv2)
        conv3 = C.layers.Convolution2D((3,3), 64)(pool2)
        f4    = C.layers.Dense(96)(conv3)
        drop4 = C.layers.Dropout(0.5)(f4)
        z     = C.layers.Dense(num_output_classes, activation=None)(drop4)

    ce = C.losses.cross_entropy_with_softmax(z, label_var)
    pe = C.metrics.classification_error(z, label_var)

    reader_train = create_reader(os.path.join(data_path, 'mnist_train.txt'), True, input_dim, num_output_classes)
    reader_test = create_reader(os.path.join(data_path, 'mnist_test.txt'), False, input_dim, num_output_classes)

    # Set learning parameters
    lr_per_sample    = [0.001]*10 + [0.0005]*10 + [0.0001]
    lr_schedule      = C.learning_rate_schedule(lr_per_sample, C.learners.UnitType.sample, epoch_size)
    mm_time_constant = [0]*5 + [1024]
    mm_schedule      = C.learners.momentum_as_time_constant_schedule(mm_time_constant, epoch_size)

    # Instantiate the trainer object to drive the model training
    local_learner = C.learners.momentum_sgd(z.parameters, lr_schedule, mm_schedule)
    parameter_learner = C.train.distributed.block_momentum_distributed_learner(local_learner, block_size=block_size)
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    trainer = C.Trainer(z, (ce, pe), parameter_learner, progress_printer)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var : reader_train.streams.features,
        label_var : reader_train.streams.labels
    }

    C.logging.log_number_of_parameters(z) ; print()

    C.train.training_session(
        trainer = trainer,
        mb_source = reader_train,
        model_inputs_to_streams = input_map,
        mb_size = minibatch_size,
        progress_frequency=epoch_size,
        max_samples = max_epochs * epoch_size,
        test_config = C.train.TestConfig(reader_test, minibatch_size=minibatch_size)
    ).train()

    # Must call MPI finalize when process exit without exceptions
    C.train.distributed.Communicator.finalize()

if __name__=='__main__':
    convnet_mnist()
