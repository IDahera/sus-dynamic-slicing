import tensorflow as tf
import keras 
import global_variables as gv
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.datasets import mnist, fashion_mnist

# Model identifiers
all_models = [gv.mnist_dense_model, gv.mnist_conv_model, gv.fashion_dense_model, gv.fashion_conv_model]
dense_models = [gv.mnist_dense_model, gv.fashion_dense_model]
conv_models = [gv.mnist_conv_model, gv.fashion_conv_model]
mnist_models = [gv.mnist_dense_model, gv.mnist_conv_model]
fashion_models = [gv.fashion_dense_model, gv.fashion_conv_model]

def get_predefined_dense_model():
    """ Creates and returns a predefined dense layer model """
    
    model = tf.keras.models.Sequential([
                Flatten(),
                Dense(50, activation='relu'),
                Dense(40, activation='relu'),
                Dense(30, activation='relu'),
                Dense(20, activation='relu'),
                Dense(10, activation='softmax')
            ])
    
    return model

def get_predefined_conv_model(shape):
    """ Creates and returns a predefined convolutional layer model. """
    
    model = tf.keras.models.Sequential([
                Conv2D(10, 2, activation='relu', input_shape=shape),
                Conv2D(10, 2, activation='relu'),
                Conv2D(10, 2, activation='relu'),
                MaxPool2D(),
                Flatten(),
                Dense(10, activation='softmax')
            ])
    
    return model

def get_predefined_model_path(model_nr):
    """ Returns the predefined model path for a given model id. """
    
    if model_nr == 1:
        model_path = "trained_models/dense_mnist.h5"
    elif model_nr == 2:
        model_path = "trained_models/conv_mnist.h5"
    elif model_nr == 3:
        model_path = "trained_models/dense_fashion_mnist.h5"
    elif model_nr == 4:
        model_path = "trained_models/conv_fashion_mnist.h5"
    else:
        raise Exception("Unknown model number.")
    
    return model_path

def load_model( model_nr,
                force_retrain=False, 
                dataset_portion=1.0, 
                epochs=1
                ):
        """ Returns a Tensorflow model trained on a selected dataset corresponding to the given model number. Access global variables for the information on the models. Dataset portion and epochs define restrictions on the training process. """
    
        # Model 1 and 2 are both trained on MNIST
        if model_nr in [1, 2]:
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Model 3 and 4 are both trained on FASHION MNIST
        elif model_nr in [3, 4]:
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        else:
            raise Exception("Unknown model number.")
            
        # MNIST and FASHION MNIST share the same shape
        shape = (28, 28, 1)
        
        # normalization
        x_train, x_test = x_train / 255.0, x_test / 255.0
        elements = int(dataset_portion * len(x_train))

        # Set path to selected model.
        model_path = get_predefined_model_path(model_nr)
        
        model_exists = True
        
        # If model exists and desired, load model. Otherwise, create and train new model.
        if not force_retrain:
            try: 
                model = keras.models.load_model(model_path)
            except:
                model_exists = False
                
        # Model 1 and 3 are both the same dense layer model
        if model_nr in [1, 3] and (not model_exists or force_retrain):
            model = get_predefined_dense_model()
            
        # Model 2 and 4 are both the same convolutional layer model
        elif model_nr in [2, 4] and (not model_exists or force_retrain):
            model = get_predefined_conv_model(shape)
        # Other models cannot be selected
        elif model_nr not in [1, 2, 3, 4]:
            raise Exception(f"Unknown model number: {model_nr}, Path: {model_path}")
        
        # The actual training process
        if not model_exists or force_retrain:
            model.compile(loss=SparseCategoricalCrossentropy(), 
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])

            model.fit(x_train[:elements], y_train[:elements], epochs=epochs)

            model.save(model_path)
        
        return model, x_train[:elements], y_train[:elements], x_test, y_test
    