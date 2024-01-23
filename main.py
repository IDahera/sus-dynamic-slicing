import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import numpy as np
import pickle
import time
import keras
from objective_function import pso_wrapper_simple, pso_wrapper_extended, get_pruned_model_acc, TrainedModel, evaluate_accuracy
import global_variables as gv
from suspiciousness import HitSpectrum, Suspiciousness
from load_model import load_model
from pyswarms.single.global_best import GlobalBestPSO

    
def get_all_layer_names(model):
    """ Given a model, returns a list containing its layers names.
    """
    return list(map(lambda x: x.name, model.layers))

def get_intermediate_output(model, l_name, sample):
    """ Given a model, a layer's name and a sample returns the corresponding layer's activation values upon receiving sample as input.
    """
    
    intermediate_output = keras.models.Model(model.input, model.get_layer(l_name).output)
    return intermediate_output(sample)

def get_element_by_index(lst, index_tuple):
    """ Returns element in the given list "lst" with the specified index. """
    
    element = lst
    for index in index_tuple:
        element = element[index]
    return element    

def main():
    if gv.APPROACH.lower() not in (gv.P_ID + gv.T_ID):
        raise Exception(
            f"Unknown approach: {gv.APPROACH}. \n Select either \"percentage\" or \"threshold\".")

    tf.random.set_seed(gv.SEED)

    # Load model
    model = None
    model, x_train, y_train, x_test, y_test = load_model(gv.MODEL_NR,
                                                         force_retrain=gv.FORCE_TRAIN,
                                                         dataset_portion=gv.DATASET_PORTION)

    x_improve = x_train[len(x_train) - len(x_test)//2:]
    y_improve = y_train[len(y_train) - len(y_test)//2:]

    # Get layer names
    l_names = get_all_layer_names(model)

    # Set batch sizes, evaluate original model on the test dataset
    batch_size = 128
    y_pred = model.predict(x_test)
    results = tf.math.argmax(y_pred, -1)
    results = tf.where(tf.math.equal(results, y_test), 1, 0)

    # Setting
    setting = TrainedModel(model,
                      x_train,
                      y_train,
                      x_improve,
                      y_improve,
                      x_test,
                      y_test)

    # Prepare batches
    ds = tf.data.Dataset.from_tensor_slices(x_improve)

    # Initialize arrays for suspiciousness scores
    all_sus_values = []
    all_sus_values_flat = []
    shapes = []
    

    if os.path.exists(gv.SUS_FILE):
        with open(gv.SUS_FILE, 'rb') as file:
            scores = pickle.load(file)
    else:
        for l_name in l_names[:-1]:
            print(f"Determine sus. values for layer {l_name} ...")

            int_output = get_intermediate_output(
                model, l_name, tf.expand_dims(x_test[0], 0))
            output_shape = int_output[0].shape
            sus = HitSpectrum(output_shape)

            for batch_index, batch in enumerate(ds.batch(batch_size, num_parallel_calls=4)):
                # Get output shape for an arbitrary sample
                int_output = get_intermediate_output(model, l_name, batch)

                # If an output is greater than 0, it is mapped to 1, and it is mapped to 0 otherwise.
                int_output = tf.where(tf.math.greater(int_output, 0), 1, 0)

                shape = int_output[0].shape
                sus.increment_as(
                    results[batch_index*batch_size:(batch_index+1)*batch_size], int_output)
                sus.increment_af(
                    results[batch_index*batch_size:(batch_index+1)*batch_size], int_output)
                sus.increment_ns(
                    results[batch_index*batch_size:(batch_index+1)*batch_size], int_output)
                sus.increment_nf(
                    results[batch_index*batch_size:(batch_index+1)*batch_size], int_output)

            a_s, a_f, n_s, n_f = sus.get_values()
            if gv.SUS_METRIC.lower() == "ochiai":
                all_sus_values.append(sus.get_ochiai())
            elif gv.SUS_METRIC.lower() == "tarantula":
                all_sus_values.append(sus.get_tarantula())
            elif gv.SUS_METRIC.lower() == f"d{gv.STAR}":
                all_sus_values.append(sus.get_d_star(gv.STAR))
            else:
                raise Exception("No support for the suspiciousness metric \"{gv.SUS_METRIC}\"")
                

            shape = all_sus_values[-1].shape
            total_elements = tf.reduce_prod(shape)
            all_sus_values_flat.append(tf.reshape(
                all_sus_values[-1], [total_elements]))

            shapes.append(shape)

        print(f"shapes: {shapes}")

        del a_s, a_f, n_s, n_f

        scores = Suspiciousness(all_sus_values,
                                         all_sus_values_flat,
                                         shapes)

        os.makedirs(os.path.dirname(gv.SUS_FILE), exist_ok=True)
        with open(gv.SUS_FILE, 'wb') as file:
            pickle.dump(scores, file)
    
    original_corr_classified, original_accuracy = evaluate_accuracy(model, x_test, y_test)
    
    if gv.APPROACH in gv.P_ID:
        # percentage bounds
        param_lower = gv.PERCENTAGE_LOWER
        param_upper = gv.PERCENTAGE_UPPER
    elif gv.APPROACH in gv.T_ID:
        # threshold bounds
        if gv.SUS_METRIC.lower() == "ochiai":
            param_lower = gv.OCHIAI_THRESHOLD_LOWER
            param_upper = gv.OCHIAI_THRESHOLD_UPPER
        elif gv.SUS_METRIC.lower() == "tarantula":
            param_lower = gv.TARANTULA_THRESHOLD_LOWER
            param_upper = gv.TARANTULA_THRESHOLD_UPPER
        else:
            raise Exception("No threshold-based pruning support for the suspiciousness metric \"{gv.SUS_METRIC}\"")

    if gv.OPTIMIZE_PRUNING:
        pruning_lower = gv.PRUNING_LOWER
        pruning_upper = gv.PRUNING_UPPER
        
        strengthening_lower = gv.STRENGTHENING_LOWER
        strengthening_upper = gv.STRENGTHENING_UPPER
        
        
        lower_bounds = [0, param_lower, pruning_lower, strengthening_lower]
        upper_bounds = [len(scores.get_sus_values()), param_upper, pruning_upper, strengthening_upper]
        optimizer = GlobalBestPSO(n_particles=gv.PSO_PARTICLES, 
                                dimensions=4, 
                                options=gv.PSO_OPTIONS,
                                bounds=(lower_bounds, upper_bounds),
                                ftol=gv.PSO_TERMINATION,
                                ftol_iter=gv.PSO_TERMINATION_ITER)
    
        best_position, best_value = optimizer.optimize(pso_wrapper_extended, 
                                                iters=gv.PSO_ITER, 
                                                setting=setting,
                                                scores=scores,
                                                approach=gv.APPROACH)
            

    else:    
        lower_bounds = [0, param_lower]
        upper_bounds = [len(scores.get_sus_values()), param_upper]
        optimizer = GlobalBestPSO(n_particles=gv.PSO_PARTICLES, 
                                dimensions=2, 
                                options=gv.PSO_OPTIONS,
                                bounds=(lower_bounds, upper_bounds),
                                ftol=gv.PSO_TERMINATION,
                                ftol_iter=gv.PSO_TERMINATION_ITER)
    
        best_position, best_value = optimizer.optimize(pso_wrapper_simple, 
                                                   iters=gv.PSO_ITER, 
                                                   setting=setting,
                                                   scores=scores,
                                                   approach=gv.APPROACH,
                                                   strengthening_factor=gv.STRENGTHENING_FACTOR,
                                                   pruning_factor=gv.PRUNING_FACTORy)
        
        
    # Log data
    layer_to_prune = int(best_value[0])
    best_param_val = best_value[1]
    pruning_factor = best_value[2] if gv.OPTIMIZE_PRUNING else gv.PRUNING_FACTOR
    strength_factor = best_value[3] if gv.OPTIMIZE_PRUNING else gv.STRENGTHENING_FACTOR
    pruned_model = get_pruned_model_acc(layer_to_prune, best_param_val, pruning_factor, strength_factor, gv.APPROACH, setting, scores)
    
    if gv.POST_TRAIN:
        model.fit(x_train, y_train, epochs=1)
        pruned_model.fit(x_train[:len(x_train) - len(x_test)], y_train[:len(y_train) - len(y_test)], epochs=1)
    
    original_corr_classified, original_accuracy = evaluate_accuracy(model, x_test, y_test)
    new_corr_classified, new_acc = evaluate_accuracy(pruned_model, x_test, y_test)
    
    old_corr_classified_train, old_acc_train = evaluate_accuracy(model, x_train, y_train)
    new_corr_classified_train, new_acc_train = evaluate_accuracy(pruned_model, x_train, y_train)
    
    print("Overview: ")
    print(f"original accuracy (test): {original_accuracy}")
    print(f"new accuracy (test): {new_acc}")
    print(f"original accuracy (train): {old_acc_train}")
    print(f"new accuracy (train): {new_acc_train}")
    print(f"original reference (test): {model.evaluate(x_test, y_test)}")
    print(f"original reference (train): {model.evaluate(x_train, y_train)}")

    corrections = new_corr_classified - original_corr_classified
    corrections_train = new_corr_classified_train - old_corr_classified_train
    
    os.makedirs(os.path.dirname(gv.RESULTS_FILE), exist_ok=True)
    print(f"Results:\n{gv.DESCRIPTION}")
    with open(gv.RESULTS_FILE, "a") as file:
        
        file.write(f"Timestamp: {time.ctime(time.time())}")
        file.write(gv.DESCRIPTION)
        file.write(f"Training Samples: {len(x_train)}, Repair Samples: {len(x_improve)}, Testing Samples: {len(x_test)}")
        file.write(f"\nstrength factor: {strength_factor}")
        file.write(f"\npruning factor: {pruning_factor}")
        file.write(f"\nbest layer: {layer_to_prune}")
        file.write(f"\nbest {gv.APPROACH} value: {best_value[1]}")
        file.write(f"\nbest pos: {best_position}")
        file.write(f"\n(Test) (2E) vs (1E + FL + 1E) accuracy: {original_accuracy} vs {new_acc} ({corrections} corrections)")
        file.write(f"\n(Train) (2E) vs (1E + FL + 1E) accuracy: {old_acc_train} vs {new_acc_train} ({corrections_train} corrections)")
        file.write("\n\n")

    del all_sus_values_flat, all_sus_values, setting

main()
