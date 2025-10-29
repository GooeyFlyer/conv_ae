# conv_ae

A convolutional autoencoder designed for anomaly detection in multivariate time-series data.

## Usage

### Option 1: Train on and detect anomalies on one file:
- Move 1 _.csv_ file into the data folder.
- Open _configuration.yml_ and change your _train_file_path_.
- Leave _test_data_config_ empty, (or same file path as _train_file_path_).

### Option 2: Train on one file, anomaly detection on another file
- Move 2 _.csv_ files into the data folder.
- Open _configuration.yml_ and change _train_file_path_ to a .csv file for training.
- Change _test_data_config_ to another .csv file for anomaly detection.

### Option 3: Split data in one file into training and anomaly detection
- Move 1 _.csv_ file into the data folder.
- Open _configuration.yml_ and change _train_file_path_ to a .csv file for training.
- Change _test_data_config_ to an index value, where you want the model to train
on data before the index, and detect anomalies on data after the index.

**Note:** the index set may not be the index used to split. It will decrease the index
(adding data to the anomaly detection split) until the length of both splits are divisible by 12.
(the number of neurons in the input for the model).

### Run the program

Change any other settings in _configuration.yml_ 

Run `python conv_ae.py`

Anomalies are listed in *anomaly_stats.txt*

Reconstructed graphs of the anomaly detection split 
and stats of the model are saved in *images/*
