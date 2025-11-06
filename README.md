# conv_ae

A convolutional autoencoder designed for anomaly detection in multivariate time-series data.

## Usage

### Parameter filtering

To filter the parameters (a.k.a. columns/features) in your *.csv* file, you can select
options for the System, Subsystem, Component, etc, in *configuration.yml*.

This requires a "lookup table", such as the one describing parameters of RCP systems.
This table must be converted into *.csv*.

### Splitting

#### Option 1: Train on and detect anomalies on one file:
- Move 1 _.csv_ file into the data folder.
- Open _configuration.yml_ and change your _train_file_path_.
- Leave _test_data_config_ empty, (or same file path as _train_file_path_).

#### Option 2: Train on one file, anomaly detection on another file
- Move 2 _.csv_ files into the data folder.
- Open _configuration.yml_ and change _train_file_path_ to a .csv file for training.
- Change _test_data_config_ to another .csv file for anomaly detection.

#### Option 3: Split data in one file into training and anomaly detection
- Move 1 _.csv_ file into the data folder.
- Open _configuration.yml_ and change _train_file_path_ to a .csv file for training.
- Change _test_data_config_ to a file line number, where you want the model to train
on data up to and including the line, and detect anomalies on data after the line.

**Note:** if a split does not fit the input shape of the model, the final datapoint
will be extended until it does.

## Run the program

Change any other settings in _configuration.yml_ 

Run `python main.py`

Anomalies are listed in *anomaly_stats.txt*

Reconstructed graphs of both the training and anomaly detection split are saved in
*images/plots/*

Stats of the model, including loss in reconstructed data,
are saved in *images/stats/*
