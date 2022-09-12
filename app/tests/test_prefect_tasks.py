"""testing module for train functions"""

import numpy as np
import pandas as pd
import prefect_tasks


def test_split_data():
    """
    Test the split_data task.
    """
    input_data = [
        (2, 6500, 800, 'Carpet Area', 'Mumbai', 'Unfurnished', 'Bachelors', 2, 1, 5),
        (3, 10000, 1200, 'Super Area', 'Kolkata', 'Furnished', 'Bachelors/Family', 2, 0, 6),
        (5, 15000, 1500, 'Super Area', 'Chennai', 'Semi-Furnished', 'Family', 3, 2, 3),
        (2, 9000, 800, 'Carpet Area', 'Bangalore ', 'Semi-Furnished', 'Bachelors', 1, -1, 5),
        (4, 13000, 1350, 'Super Area', 'Mumbai', 'Semi-Furnished', 'Bachelors/Family', 2, 5, 5),
        (3, 8000, 800, 'Carpet Area', 'Delhi', 'Semi-Furnished', 'Family', 1, 6, 12),
        (3, 8500, 850, 'Super Area', 'Chennai', 'Unfurnished', 'Bachelors', 2, 3, 10),
        (5, 12000, 1300, 'Carpet Area', 'Hyderabad', 'Furnished', 'Bachelors/Family', 2, 4, 25)
    ]

    input_columns = [
        'BHK',
        'Rent',
        'Size',
        'Area Type',
        'City',
        'Furnishing Status',
        'Tenant Preferred',
        'Bathroom',
        'Floor',
        'Total Floors',
    ]

    input_df = pd.DataFrame(input_data, columns=input_columns)

    df_train, df_val = prefect_tasks.split_data.fn(input_df)

    actual_df_train_shape = df_train.shape
    actual_df_val_shape = df_val.shape

    expected_df_train_shape = (6, 10)
    expected_df_val_shape = (2, 10)

    assert actual_df_train_shape == expected_df_train_shape
    assert actual_df_val_shape == expected_df_val_shape


def test_prepare_data():
    """
    Tests the prepare_data task.
    """
    input_train_data = [
        (2, 6500, 800, 'Carpet Area', 'Mumbai', 'Unfurnished', 'Bachelors', 2, 1, 5),
        (3, 10000, 1200, 'Super Area', 'Kolkata', 'Furnished', 'Bachelors/Family', 2, 0, 6),
        (5, 15000, 1500, 'Super Area', 'Chennai', 'Semi-Furnished', 'Family', 3, 2, 3),
        (2, 9000, 800, 'Carpet Area', 'Bangalore ', 'Semi-Furnished', 'Bachelors', 1, -1, 5),
        (2, 7000, 750, 'Super Area', 'Delhi', 'Unfurnished', 'Family', 1, 1, 3),
        (5, 12000, 1300, 'Carpet Area', 'Hyderabad', 'Semi-Furnished', 'Bachelors/Family', 2, 4, 25)
    ]

    input_val_data = [
        (4, 13000, 1350, 'Super Area', 'Mumbai', 'Furnished', 'Bachelors/Family', 2, 5, 5),
        (3, 8000, 800, 'Carpet Area', 'Delhi', 'Semi-Furnished', 'Family', 1, 6, 12),
    ]

    input_columns = [
        'BHK',
        'Rent',
        'Size',
        'Area Type',
        'City',
        'Furnishing Status',
        'Tenant Preferred',
        'Bathroom',
        'Floor',
        'Total Floors',
    ]

    df_train = pd.DataFrame(input_train_data, columns=input_columns)
    df_val = pd.DataFrame(input_val_data, columns=input_columns)

    target = 'Rent'

    actual_X_train, actual_X_val, actual_y_train, actual_y_val, _ = prefect_tasks.prepare_data.fn(df_train, df_val, input_columns, target)

    expected_X_train = np.array(
        [
            [1.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 800.0, 1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1200.0, 0.0, 1.0, 0.0, 6.0],
            [0.0, 1.0, 5.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1500.0, 0.0, 0.0, 1.0, 3.0],
            [1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 800.0, 1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 750.0, 0.0, 0.0, 1.0, 3.0],
            [1.0, 0.0, 5.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0, 1300.0, 0.0, 1.0, 0.0, 25.0]
        ],
        dtype=float,
    )

    expected_X_val = np.array(
        [
            [0.0, 1.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 1.0, 0.0, 0.0, 1350.0, 0.0, 1.0, 0.0, 5.0],
            [1.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 1.0, 0.0, 800.0, 0.0, 0.0, 1.0, 12.0]
        ],
        dtype=float,
    )

    expected_y_train = np.array([6500, 10000, 15000, 9000, 7000, 12000])

    expected_y_val = np.array([13000, 8000])

    assert (actual_X_train == expected_X_train).all()
    assert (actual_X_val == expected_X_val).all()
    assert (actual_y_train == expected_y_train).all()
    assert (actual_y_val == expected_y_val).all()