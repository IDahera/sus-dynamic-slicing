# Neural Network Pruning Based on Spectrum-Based Fault Localization and Particle Swarm Optimization

This repository contains the ``Python 2.10`` software artifacts for the paper "Neural Network Pruning Based on Spectrum-Based Fault Localization and Particle Swarm Optimization". The following python libraries must be installed to run the application.
- Tensorflow 2.9: ``pip install tensorflow==2.9``
- Pickle: ``pip install pickle``
- PySwarms: ``pip install pyswarms``
- Numpy: ``pip install numpy``

## Quick Start
Run the following command to run the application: ``python main.py``

## Files
| Name | Description |
| -------- | ------- |
| ``load_model.py`` | Responsible for creating, buffering and loading one out of 4 predefined models. |
| ``custom_layer.py`` | Defines a custom Tensorflow layer used by the application for model pruning. |
| ``global_variables.py`` | Defines a set of variable configuring the pruning process. For instance, the model used for the work's experiments can be selected. |
| ``main.py`` | Main file initiating the pruning process. |
| ``objective_function.py`` | Defines the objective functions for particle swarm optimization. |
| ``suspiciousness.py`` | An implementation of hit-spectrum analysis for neural networks. |
