import numpy as np
import tensorflow as tf
from custom_layer import CustomPruningLayer
from keras import Sequential
from global_variables import T_ID, P_ID, DEBUGGING
from suspiciousness import Suspiciousness


class TrainedModel():
    def __init__(self,
                 model,
                 x_train,
                 y_train,
                 x_improve,
                 y_improve,
                 x_test,
                 y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_improve = x_improve
        self.y_improve = y_improve
        self.x_test = x_test
        self.y_test = y_test

    def get_model(self):
        return self.model

    def get_x_train(self):
        return self.x_train

    def get_y_train(self):
        return self.y_train
    
    def get_x_improve(self):
        return self.x_improve 
    
    def get_y_improve(self):
        return self.y_improve 

    def get_x_test(self):
        return self.x_test

    def get_y_test(self):
        return self.y_test

def evaluate_accuracy(model, samples, labels) -> (int, float):
    """ Custom function obtaining a model's accuracy. """
    
    try:
        predictions = model.predict(samples, verbose=0)
        predicted_indices = np.argmax(predictions, axis=1)
        correctness = (predicted_indices == labels)
        correctly_classified = tf.reduce_sum(tf.cast(correctness, tf.int32))
        accuracy = correctly_classified / len(labels) * 100
        return correctly_classified, accuracy
    except:
        # If an error occurs, return the worst possible outcome
        return -len(samples), 0

def pso_wrapper_extended(variables, approach: str, setting: TrainedModel, scores: Suspiciousness):
    """ A wrapper function for "get_pruned_model_acc" """
    
    results = []
    for variable in variables:
        
        if DEBUGGING:
            print(f"Variable: {variable}")
        
        layer_to_prune, approach_param, pruning_factor, strengthening_factor = variable
        modified_model = get_pruned_model_acc(int(layer_to_prune), 
                                            approach_param, 
                                            pruning_factor, 
                                            strengthening_factor,
                                            approach, setting,
                                            scores)
        _, acc = evaluate_accuracy(modified_model, setting.get_x_test(), setting.get_y_test())
        results.append(-acc)
    return results

def pso_wrapper_simple(variables, pruning_factor: float, strengthening_factor: float, approach: str, setting: TrainedModel, scores: Suspiciousness):
    """ A wrapper function for "get_pruned_model_acc" """
    
    results = []
    for variable in variables:

        if DEBUGGING:
            print(f"Variable: {variable}")
        
        layer_to_prune, approach_param = variable
        modified_model = get_pruned_model_acc(int(layer_to_prune), 
                                            approach_param, 
                                            pruning_factor, 
                                            strengthening_factor,
                                            approach, setting,
                                            scores)
        _, acc = evaluate_accuracy(modified_model, setting.get_x_test(), setting.get_y_test())
        results.append(-acc)
    return results
    
def get_pruned_model_acc(layer_to_prune,
                         approach_param,
                         pruning_factor,
                         strengthening_factor,
                         approach,
                         setting: TrainedModel,
                         scores: Suspiciousness):
    """ Given a trained model stored in "setting" and tensors stored in "scores" reflecting its suspiciousness scores, adds a pruning layer to the model according to the parameters pruning a selected layer's neurons. Represents the objective function to be maximized. """

    if DEBUGGING:
        print("Start model pruning ...")

    total_neurons = len(scores.get_sus_values_flat()[layer_to_prune - 1])
    ochiai_values_flat = scores.get_sus_values_flat()[layer_to_prune - 1]
    ochiai_values = scores.get_sus_values()[layer_to_prune - 1]
    model = setting.get_model()
    shape = scores.get_shapes()[layer_to_prune - 1]
    
    if approach.lower() in P_ID:
        n = int(float(total_neurons) * approach_param)
    elif approach.lower() in T_ID:
        # A little help by ChatGPT
        mask_exceeds_threshold = ochiai_values_flat > approach_param
        mask_exceeds_threshold = tf.cast(mask_exceeds_threshold, tf.int32)
        n = int(tf.reduce_sum(mask_exceeds_threshold))
    else:
        raise Exception("An error occurred.")
    
    if DEBUGGING:
        print("Determine neurons to be pruned ...")

    values, _ = tf.math.top_k(ochiai_values_flat, k=n)
    indices = []
    for value in values:
        subindices = tf.where(ochiai_values == value)
        for idx in subindices:
            indices.append(idx)

    mod_output_shape = tuple(
        1 if element is None else element for element in shape)
    
    modified_model = Sequential()

    for i in range(layer_to_prune):
        modified_model.add(model.layers[i])
        modified_model.layers[-1].set_weights(model.layers[i].get_weights())
        
    # Create and insert mask layer
    mask = tf.Variable(tf.ones(mod_output_shape), dtype=tf.float32)
    mask = mask + strengthening_factor

    if n > 0 and len(indices) > 0:
        mask = tf.tensor_scatter_nd_update(
            mask, indices, np.repeat(pruning_factor, len(indices)))
    custom_layer = CustomPruningLayer(mask)
    custom_layer.trainable = False
    modified_model.add(custom_layer)  

    # Add the remaining layers after the custom layer
    for i in range(layer_to_prune, len(model.layers)):
        modified_model.add(model.layers[i])
        modified_model.layers[-1].set_weights(model.layers[i].get_weights())
        
    # Compile model
    modified_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])
    
    return modified_model