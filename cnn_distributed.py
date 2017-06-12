"""
CNTK Tutorials
103 Part D: Convolutional Neural Network with MNIST
https://notebooks.azure.com/cntk/libraries/tutorials/html/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.ipynb
"""
import numpy as np
import os
import sys
import time
import cntk

# Ensure we always get the same amount of randomness
np.random.seed(0)

# Define the data dimensions
input_dim_model = (1, 28, 28)    # images are 28 x 28 with 1 channel of color (gray)
input_dim = 28*28                # used by readers to treat input data as a vector
num_output_classes = 10

# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):

    ctf = cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(
          labels=cntk.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
          features=cntk.io.StreamDef(field='features', shape=input_dim, is_sparse=False)))

    return cntk.io.MinibatchSource(ctf,
        randomize = is_training, max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)

# function to build model
def create_model(features):
    with cntk.layers.default_options(init = cntk.layers.glorot_uniform(), activation = cntk.relu):
            h = features
            h = cntk.layers.Convolution2D(filter_shape=(5,5), num_filters=8, strides=(1,1), pad=True, name="first_conv")(h)
            h = cntk.layers.MaxPooling(filter_shape=(2,2), strides=(2,2), name="first_max")(h)
            h = cntk.layers.Convolution2D(filter_shape=(5,5), num_filters=16, strides=(1,1), pad=True, name="second_conv")(h)
            h = cntk.layers.MaxPooling(filter_shape=(3,3), strides=(3,3), name="second_max")(h)
            r = cntk.layers.Dense(num_output_classes, activation = None, name="classify")(h)
            return r

x = cntk.input(input_dim_model)
y = cntk.input(num_output_classes)
z = create_model(x)


# Training

def create_criterion_function(model, labels):
    loss = cntk.cross_entropy_with_softmax(model, labels)
    errs = cntk.classification_error(model, labels)
    return loss, errs # (model, labels) -> (loss, error metric)

# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=5):
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))

    return mb, training_loss, eval_error

def train_test(train_reader, test_reader, model_func, num_sweeps_to_train_with=10):

    # Instantiate the model function; x is the input (feature) variable
    # We will scale the input image pixels within 0-1 range by dividing all input value by 255.
    model = model_func(x/255)

    # Instantiate the loss and error function
    loss, label_error = create_criterion_function(model, y)

    # Instantiate the trainer object to drive the model training
    learning_rate = 0.2
    lr_schedule = cntk.learning_rate_schedule(learning_rate, cntk.UnitType.minibatch)
    learner = cntk.sgd(z.parameters, lr_schedule)
    trainer = cntk.Trainer(z, (loss, label_error), [learner])

    # Initialize the parameters for the trainer
    minibatch_size = 64
    num_samples_per_sweep = 60000
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

    # Map the data streams to the input and labels.
    input_map={
        y  : train_reader.streams.labels,
        x  : train_reader.streams.features
    }

    # Uncomment below for more detailed logging
    training_progress_output_freq = 500

    # Start a timer
    start = time.time()

    for i in range(0, int(num_minibatches_to_train)):
        # Read a mini batch from the training data file
        data=train_reader.next_minibatch(minibatch_size, input_map=input_map)
        trainer.train_minibatch(data)
        print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

    # Print training time
    print("Training took {:.1f} sec".format(time.time() - start))

    # Test the model
    test_input_map = {
        y  : test_reader.streams.labels,
        x  : test_reader.streams.features
    }

    # Test data for trained model
    test_minibatch_size = 512
    num_samples = 10000
    num_minibatches_to_test = num_samples // test_minibatch_size

    test_result = 0.0

    for i in range(num_minibatches_to_test):

        # We are loading test data in batches specified by test_minibatch_size
        # Each data point in the minibatch is a MNIST digit image of 784 dimensions
        # with one pixel per dimension that we will encode / decode with the
        # trained model.
        data = test_reader.next_minibatch(test_minibatch_size, input_map=test_input_map)
        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))

if __name__ == '__main__':
    data_dir = "data"
    train_file = os.path.join(data_dir, "mnist_train.txt")
    test_file = os.path.join(data_dir, "mnist_test.txt")

    reader_train = create_reader(train_file, True, input_dim, num_output_classes)
    reader_test = create_reader(test_file, False, input_dim, num_output_classes)

    train_test(reader_train, reader_test, z)
