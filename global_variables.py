import numpy as np

# Global variable not to be modified 
SEED = 42
PSO_ITER = 50
PSO_PARTICLES = 5
PSO_OPTIONS = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
PERCENTAGE_LOWER = 0.0
PERCENTAGE_UPPER = 0.1
OCHIAI_THRESHOLD_LOWER = 0.5
OCHIAI_THRESHOLD_UPPER = 1.0
TARANTULA_THRESHOLD_LOWER = 0.5
TARANTULA_THRESHOLD_UPPER = 1.0
STAR = 1
PRUNING_LOWER = 0.0
PRUNING_UPPER = 1.0
STRENGTHENING_LOWER = 1.0
STRENGTHENING_UPPER = 10.0

P_ID = ["percentage", "percentage-based", "percentage_based"]
T_ID = ["threshold", "threshold-based", "threshold_based"]
SUS_METRICS = ["ochiai", "tarantula", f"d{STAR}"]

# Global variables to be set manually
MODEL_NR = 4
OPTIMIZE_PRUNING = True
DEBUGGING = False
FORCE_TRAIN = False
DATASET_PORTION = 1.0
APPROACH = T_ID[0]
SUS_METRIC = SUS_METRICS[1]
__PSO_EARLY_TERMINATION = True
POST_TRAIN = True

# Global variables to be set automatically
if MODEL_NR in [1, 3]:
    PRUNING_FACTOR = 0.95
    STRENGTHENING_FACTOR = 1.1
elif MODEL_NR in [2, 4]:
    PRUNING_FACTOR = 0.0
    STRENGTHENING_FACTOR = 1.1

# Models' IDs
mnist_dense_model = 1
mnist_conv_model = 2
fashion_dense_model = 3
fashion_conv_model = 4

if MODEL_NR == 1:
    __MODEL_DESCR = "dense_mnist"
elif MODEL_NR == 2:
    __MODEL_DESCR = "conv_mnist"
elif MODEL_NR == 3:
    __MODEL_DESCR = "dense_fashion"
elif MODEL_NR == 4:
    __MODEL_DESCR = "conv_fashion"

PSO_TERMINATION = 1 if __PSO_EARLY_TERMINATION else -np.inf
PSO_TERMINATION_ITER = 3

# Preamble for results files
SUS_FILE = f"buffer/sus_{__MODEL_DESCR}_{SUS_METRIC}.pickle"
RESULTS_FILE = f"results/{APPROACH}/{MODEL_NR}_pso_{__MODEL_DESCR}.txt"
DESCRIPTION=f"""
### Model: {__MODEL_DESCR}, 
### Variable pruning/strength?: {OPTIMIZE_PRUNING}, 
### SUS. Metric: {SUS_METRIC}, 
### Dataset: {DATASET_PORTION * 100}%
### PSO: {PSO_PARTICLES} particles / {PSO_ITER} iterations, Early termination? {__PSO_EARLY_TERMINATION}
### POST Train: {POST_TRAIN} """