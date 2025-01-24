"""
This code is based on the open-source project:
https://github.com/guoshnBJTU/ASTGCN-2019-pytorch
"""
import os
from lib import config_loader
import numpy as np


def search_data_idx(len_data,
                    num_depend,
                    label_start_idx,
                    num_predict,
                    units,
                    points_per_hour=12):
    """
    Generate the indices required for searching data to make predictions.

    Parameters
    ----------
    len_data : int
        The length of the entire historical data.
    num_depend : int
        The number of dependent time periods (e.g., weeks, days, hours) to use for prediction.
    label_start_idx : int
        The starting index of the target data for prediction.
    units : int
        The number of units in the time period (e.g. 7 * 24 for a week, 24 for a day, 1 for an hour).

    Returns
    -------
    list[(start_idx, end_idx)]
        A list of (start_idx, end_idx) pairs, where each tuple is of the form (int, int). These
        tuples indicate the start and end indices of the input data segments to be used for
        generating predictions.
    """
    if points_per_hour <= 0:
        raise ValueError("points_per_hour must be greater than 0.")

    if label_start_idx + num_predict > len_data:
        return None

    data_idx = []
    for i in range(1, num_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_predict
        if start_idx >= 0:
            data_idx.append((start_idx, end_idx))
        else:
            return None

    if len(data_idx) != num_depend:
        return None

    return data_idx[::-1]


def get_sample_data(data_seq,
                    num_weeks,
                    num_days,
                    num_hours,
                    label_start_idx,
                    num_predict,
                    points_per_hour=12):
    """
    Retrieve data based on idx.

    Parameters
    ----------
    data_seq : np.ndarray
        The input data sequence with shape (len_data, num_vertices, num_features).
    label_start_idx : int
        The starting index of the target data for prediction.

    Returns
    -------
    (week_sample, day_sample, hour_sample, target)
        week_sample.shape = (num_weeks * points_per_hour, num_vertices, num_features).
        day_sample.shape = (num_days * points_per_hour, num_vertices, num_features).
        hour_sample.shape = (num_hours * points_per_hour, num_vertices, num_features).
        target.shape = (num_predict, num_vertices, num_features).
    """
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_predict > data_seq.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_weeks > 0:
        week_indices = search_data_idx(data_seq.shape[0],
                                       num_weeks,
                                       label_start_idx,
                                       num_predict,
                                       7 * 24,
                                       points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_seq[i: j] for i, j in week_indices], axis=0)

    if num_days > 0:
        day_indices = search_data_idx(data_seq.shape[0],
                                      num_days,
                                      label_start_idx,
                                      num_predict,
                                      24,
                                      points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_seq[i: j] for i, j in day_indices], axis=0)

    if num_hours > 0:
        hour_indices = search_data_idx(data_seq.shape[0],
                                       num_hours,
                                       label_start_idx,
                                       num_predict,
                                       1,
                                       points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_seq[i: j] for i, j in hour_indices], axis=0)

    target = data_seq[label_start_idx: label_start_idx + num_predict]

    return week_sample, day_sample, hour_sample, target


def standardization(train, val, test):
    """
    Standardize the training, validation, and test datasets.
    """
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=(0, 1, 3), keepdims=True)
    std = train.std(axis=(0, 1, 3), keepdims=True)

    def standardize(x):
        return (x - mean) / std

    train_norm = standardize(train)
    val_norm = standardize(val)
    test_norm = standardize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm


def read_and_generate_dataset(traffic_filename,
                              num_weeks,
                              num_days,
                              num_hours,
                              num_predict,
                              points_per_hour=12,
                              save=False):
    """
    Generate processed dataset.

    Parameters
    ----------
    traffic_filename : str
        The .npz file of the dataset.
    num_weeks : int
        The number of weeks to predict.
    num_days : int
        The number of days to predict.
    num_hours : int
        The number of hours to predict.
    num_predict : int
        The number of points to predict for each sample.
    points_per_hour : int, optional
        The number of data points per hour. Default is 12.
    save : bool, optional
        Do you want to save the processed dataset.Default is False.
    """
    # (len_data, num_vertices, num_features)
    data_seq = np.load(traffic_filename)['data']

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_data(data_seq,
                                 num_weeks,
                                 num_days,
                                 num_hours,
                                 idx,
                                 num_predict,
                                 points_per_hour)

        if (sample[0] is None) and (sample[1] is None) and (sample[2] is None):
            continue

        week_sample, day_sample, hour_sample, target = sample

        # [(week_sample), (day_sample), (hour_sample), target, time_sample]
        sample = []

        # (batch_size, num_sensors, features, time_steps)
        # (1, N, F, T)
        if num_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))
            sample.append(week_sample)

        if num_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))
            sample.append(day_sample)

        if num_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))
            sample.append(hour_sample)

        # (1, N, T)
        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
        sample.append(target)

        # (1, idx)
        time_sample = np.expand_dims(np.array([idx]), axis=0)
        sample.append(time_sample)

        # sample：[(week_sample), (day_sample), (hour_sample), target, time_sample] =
        #         [(1, N, F, Tw), (1, N, F, Td), (1, N, F, Th), (1, N, Tpre), (1, 1)]
        all_samples.append(sample)

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    # [(B, N, F, Tw), (B, N, F, Td), (B, N, F, Th), (B, N, Tpre), (B, idx)]
    training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line1])]
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]

    # (B, N, F, T')
    train_x = np.concatenate(training_set[:-2], axis=-1)
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    # (B, N, T)
    train_target = training_set[-2]
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    # (B, idx)
    train_timestamp = training_set[-1]
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    (stats, train_x_norm, val_x_norm, test_x_norm) = standardization(train_x, val_x, test_x)

    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_mean': stats['_mean'],
            '_std': stats['_std'],
        }
    }

    if save:
        file = dataset_name
        dirpath = os.path.dirname(traffic_filename)
        filename = os.path.join(dirpath, file + '_w' + str(num_weeks) + '_d' + str(num_days) + '_h' + str(num_hours))
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'],
                            train_target=all_data['train']['target'],
                            train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'],
                            val_target=all_data['val']['target'],
                            val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'],
                            test_target=all_data['test']['target'],
                            test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_mean'],
                            std=all_data['stats']['_std'])
        print('save file: ', filename)

    return all_data


if __name__ == '__main__':

    args = config_loader.setTerminal()
    data_config, train_config = config_loader.getConfig(args)

    # config data info
    traffic_filename = data_config['traffic_filename']
    dataset_name = data_config['dataset_name']
    points_per_hour = int(data_config['points_per_hour'])
    num_predict = int(data_config['num_predict'])
    num_weeks = int(data_config['num_weeks'])
    num_days = int(data_config['num_days'])
    num_hours = int(data_config['num_hours'])

    all_data = read_and_generate_dataset(traffic_filename,
                                         num_weeks,
                                         num_days,
                                         num_hours,
                                         num_predict,
                                         points_per_hour=points_per_hour,
                                         save=True)
