# Bayesian Uncertainty Modelling for Cloud Workload Prediction

## Introduction

Providers of cloud computing systems need to allocate resources carefully in order to (i) meet the desired Quality of Service, and (ii) reduce waste due to overallocation. To allocate resources to service requests without excessive delays, we need to predict future demand. Current state-of-the-art methods such as Long Short-Term Memory-based (LSTM) models make only point predictions of demand and ignore uncertainty. Predicting a distribution would provide a more complete picture and inform resource scheduling decisions. We investigate DL models to predict workload distribution, and evaluate it on the Time Series forecasting of CPU and memory workload of 8 clusters in the Google Cloud data centre. Experiments show that the proposed models have a similar point forecast accuracy to the LSTM. However, they can provide better estimations of resource usage bounds, allowing a reduction of both overprediction and total predicted resources, maintaining good runtime performance.

## Python Dependencies
* arch                      5.1.0
* keras                     2.8.0
* matplotlib                3.3.4
* numpy                     1.21.5
* pandas                    1.2.3
* python                    3.7.9
* statsmodels               0.12.2
* talos                     1.0.2 
* tensorflow                2.8.0
* tensorflow-gpu            2.8.0
* tensorflow-probability    0.14.0

## Project Structure
* **hyperparams**: contains for each deep learning model the list of optimal hyperparameters found with Talos.
* **img**: contains output plot for predictions, models and loss function.
* **models**: contains the definition of statistical and deep learning models. One can train the model from scratch using the optimal parameters found with Talos, look for the optimal hyperparameters by changing the search space dictionary or load a saved model and make new forecasts.
* **param**: contains for each statistical model the list of optimal parameters found.
* **res**: contains the results of the prediction
* **saved_data**: contains the preprocessed datasets.
* **saved_models**: contains the model saved during the training phase.
* **time**: contains measurements of the time for training, fine-tuning and inference phases.
* **util**: contains useful methods for initialising the datasets, plotting and saving the results.

## Statistical Methods

#### Train ARIMA

```bash
python arima_training.py
```

#### Train GARCH

```bash
python garch_training.py
```

## Deep Learning Methods

#### Train LSTM

```bash
python lstm_training.py
```

#### Train HBNN

```bash
python hbnn_training.py
```

#### Train LSTMD

```bash
python lstmd_training.py
```

