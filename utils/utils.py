try:
    import pandas as pd
    # import numpy as np
    import os
except ModuleNotFoundError:
    import sys
    print('Module Import error. Check installation')


def get_file_name(path: str = '.'):
    return path.split('\\')[-1]


def import_data_info(path: str = '.',
                     csv_file: str = 'driving_log.csv'):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, csv_file), names=columns)
    data['Center'] = data['Center'].apply(get_file_name)
    return data
