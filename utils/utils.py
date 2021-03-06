try:
    import os
    import pandas as pd
    import numpy as np
    # import matplotlib
    #
    # matplotlib.use('TkAgg')

    import matplotlib.pyplot as plt
    import matplotlib.image as mp_img
    from sklearn.utils import shuffle
    from imgaug import augmenters as aug
    from cv2 import cv2

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Convolution2D, Flatten, Dense

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
        # Index 0 Corresponds to File name
        img_path = os.path.abspath(os.path.join(path, 'IMG', i_data[0]))
        image_paths.append(img_path)
        # Index 3 Corresponds to Steering angle value
        steering_values.append(float(i_data[3]))
    image_paths = np.asarray(image_paths)
    steering_values = np.asarray(steering_values)

    return image_paths, steering_values


def read_img(img_path):
    return mp_img.imread(img_path)


def augment_image(img_path, steering):
    img = read_img(img_path)
    # PAN augmentation
    if np.random.rand() < 0.5:
        pan = aug.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})
        img = pan.augment_image(img)
    # Zoom
    if np.random.rand() < 0.5:
        zoom = aug.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    # Brightness
    if np.random.rand() < 0.5:
        brightness = aug.Multiply((0.4, 1.2))
        img = brightness.augment_image(img)
    # Flip
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    return img, steering


def pre_process(img):
    img = img[60:135, :, :]
    img = cv2.resize(img, (200, 66))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = img / 255  # Normalize the image
    return img


def create_model(params):
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation=params['activation']))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation=params['activation']))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation=params['activation']))
    model.add(Convolution2D(64, (3, 3), activation=params['activation']))
    model.add(Convolution2D(64, (3, 3), activation=params['activation']))

    model.add(Flatten())

    model.add(Dense(100, activation=params['activation']))
    model.add(Dense(50, activation=params['activation']))
    model.add(Dense(10, activation=params['activation']))
    model.add(Dense(1))

    model.compile(Adam(learning_rate=params['learning_rate']), loss=params['loss_fn'])

    return model
