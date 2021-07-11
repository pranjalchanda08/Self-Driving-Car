try:
    import pandas as pd
    import numpy as np
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from sklearn.utils import shuffle
    import os
except ModuleNotFoundError:
    import sys

    print('Module Import error. Check installation')
    sys.exit(-1)


def get_file_name(path: str = '.'):
    return path.split('\\')[-1]


def import_data_info(path: str = '.',
                     csv_file: str = 'driving_log.csv'):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, csv_file), names=columns)
    data['Center'] = data['Center'].apply(get_file_name)
    data['Left'] = data['Left'].apply(get_file_name)
    data['Right'] = data['Right'].apply(get_file_name)
    return data


def get_balance_data(data,
                     display: bool = True):
    n_bins = 31
    # Bin Cutoff value
    n_samples = 2500
    hist, bins = np.histogram(data['Steering'], n_bins)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        x_range = (n_samples, n_samples)
        y_range = (-1, 1)
        plt.bar(center, hist, width=0.06)
        plt.plot(y_range, x_range)
        plt.show()

    removed_data_list = []
    for j in range(n_bins):
        bin_data_list = []
        for i, angle in enumerate(data['Steering']):
            if bins[j] <= angle <= bins[j + 1]:
                bin_data_list.append(i)
        bin_data_list = shuffle(bin_data_list)
        # Split random samples according to cut off value
        bin_data_list = bin_data_list[n_samples:]
        removed_data_list.extend(bin_data_list)
    data.drop(data.index[removed_data_list], inplace=True)
    if display:
        hist, _ = np.histogram(data['Steering'], n_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        x_range = (n_samples, n_samples)
        y_range = (-1, 1)
        plt.bar(center, hist, width=0.06)
        plt.plot(y_range, x_range)
        plt.show()
        print('Removed data len: ', len(removed_data_list))
        print('Remaining data len: ', len(data))

    return data


def load_data(path: str, data):
    image_paths = []
    steering_values = []
    for i in range(len(data)):
        i_data = data.iloc[i]
        image_paths.append(os.path.join(path, 'IMG', i_data[0]))  # Index 0 Corresponds to File name
        steering_values.append(float(i_data[3]))  # Index 3 Corresponds to Steering angle value
    image_paths = np.asarray(image_paths)
    steering_values = np.asarray(steering_values)

    return image_paths, steering_values
