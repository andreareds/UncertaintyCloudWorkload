# Bayesian Uncertainty Modelling for Cloud Workload Prediction

## Introduction

Providers of cloud computing systems need to allocate resources carefully in order to (i) meet the desired Quality of Service, and (ii) reduce waste due to overallocation. To allocate resources to service requests without excessive delays, we need to predict future demand. Current state-of-the-art methods such as Long Short-Term Memory-based (LSTM) models make only point predictions of demand and ignore uncertainty. Predicting a distribution would provide a more complete picture and inform resource scheduling decisions. We investigate DL models to predict workload distribution, and evaluate it on the Time Series forecasting of CPU and memory workload of 8 clusters in the Google Cloud data centre. Experiments show that the proposed models have a similar point forecast accuracy to the LSTM. However, they can provide better estimations of resource usage bounds, allowing a reduction of both overprediction and total predicted resources, maintaining good runtime performance.

## Python Packages
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

