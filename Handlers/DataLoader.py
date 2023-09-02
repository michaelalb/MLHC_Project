"""
notes on the data -
there are two training sets.
each training set contains about 20k files.
each file is a subject.

"""
import glob
import random
import joblib

import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

from Consts import LABEL_COL, STATIC_COVARIANT_COLS, DYNAMIC_COVARIANT_COLS, TIME_COL


class DataLoader:

    def load_data_for_training(self, data_path: str = "./Data", should_up_sample_pos: bool = False,
                               up_sample_target_ratio: float = 0.5, should_shuffle: bool = True,
                               train_ratio: float = 0.8, scaler: Scaler = None,
                               max_amount_of_ts_to_load: int = 0) -> (
            list[TimeSeries], list[TimeSeries], list[TimeSeries], list[TimeSeries]):
        """
        load data for training and evaluation.
        :param data_path: path to the data folder.
        :param should_up_sample_pos: should up sample positive examples.
        :param up_sample_target_ratio: target ratio of positive examples.
        :param should_shuffle: should shuffle the data.
        :param train_ratio: ratio of data to be used for training.
        :param scaler: scaler to use for the data.
        :return: train_pos_label_ts, train_pos_cov_ts, train_neg_label_ts, train_neg_cov_ts
        """
        # load the data as ts and preprocess it
        pos_label_ts, pos_cov_ts, neg_label_ts, neg_cov_ts = self.load_data_as_time_series(data_path, scaler,
                                                                                           max_amount_of_ts_to_load)

        # train test split
        train_label_ts, train_cov_ts, test_label_ts, test_cov_ts = self.train_test_split(pos_label_ts, pos_cov_ts,
                                                                                         neg_label_ts, neg_cov_ts,
                                                                                         train_ratio, should_shuffle,
                                                                                         should_up_sample_pos,
                                                                                         up_sample_target_ratio)

        return train_label_ts, train_cov_ts, test_label_ts, test_cov_ts

    def shuffle(self, list1: list, list2: list) -> (list, list):
        """
        shuffle the data but gaurntees order between the two lists remains intact.
        """
        assert len(list1) == len(list2)

        indices = list(range(len(list1)))

        # Shuffle the indices
        random.shuffle(indices)

        # Use the shuffled indices to create shuffled lists
        shuffled_list1 = [list1[i] for i in indices]
        shuffled_list2 = [list2[i] for i in indices]

        return shuffled_list1, shuffled_list2

    def split_lists_by_ration(self, list1: list, list2: list, ratio: float) -> (list, list, list, list):
        """
        split two lists by a ratio.
        """
        assert len(list1) == len(list2)

        split_index = int(len(list1) * ratio)

        list1_1 = list1[:split_index]
        list1_2 = list1[split_index:]

        list2_1 = list2[:split_index]
        list2_2 = list2[split_index:]

        return list1_1, list1_2, list2_1, list2_2

    def train_test_split(self, pos_label_ts: list[TimeSeries], pos_cov_ts: list[TimeSeries],
                         neg_label_ts: list[TimeSeries], nge_cov_ts: list[TimeSeries], train_size: float = 0.8,
                         shuffle: bool = True, should_up_sample_pos: bool = False,
                         pos_up_sample_target_ratio: float = 0) -> (TimeSeries, TimeSeries, TimeSeries, TimeSeries):
        """
        split the data to train and test sets - while maintaining the order between the label and covariant series.
        :param pos_label_ts: list of positive label time series
        :param pos_cov_ts: list of positive covariant time series
        :param neg_label_ts: list of negative label time series
        :param nge_cov_ts: list of negative covariant time series
        :param train_size: the size of the test set
        :param shuffle: should shuffle the data
        :param should_up_sample_pos: should up sample the positive data
        :param pos_up_sample_target_ratio: the target ratio between the positive and negative data
        :return: train_label_ts, train_cov_ts, test_label_ts, test_cov_ts
        """
        if shuffle:
            pos_label_ts, pos_cov_ts = self.shuffle(pos_label_ts, pos_cov_ts)
            neg_label_ts, nge_cov_ts = self.shuffle(neg_label_ts, nge_cov_ts)

        if should_up_sample_pos:
            current_ratio = len(pos_label_ts) / len(neg_label_ts)
            assert (current_ratio < pos_up_sample_target_ratio, "current ratio is higher than target ratio")
            # calculate how many to add
            how_many_to_add = int(len(neg_label_ts) * (pos_up_sample_target_ratio - current_ratio))
            idx_to_add = random.choices(range(len(pos_label_ts)), k=how_many_to_add)
            # add the data to the time series list
            pos_label_ts = pos_label_ts + [pos_label_ts[i] for i in idx_to_add]
            pos_cov_ts = pos_cov_ts + [pos_cov_ts[i] for i in idx_to_add]

        # split the data
        # positive
        pos_train_label_ts, pos_test_label_ts, pos_train_cov_ts, pos_test_cov_ts = self.split_lists_by_ration(
            pos_label_ts, pos_cov_ts, train_size)

        # negative
        neg_train_label_ts, neg_test_label_ts, neg_train_cov_ts, neg_test_cov_ts = self.split_lists_by_ration(
            neg_label_ts, nge_cov_ts, train_size)

        # merge the data
        train_label_ts = pos_train_label_ts + neg_train_label_ts
        train_cov_ts = pos_train_cov_ts + neg_train_cov_ts
        test_label_ts = pos_test_label_ts + neg_test_label_ts
        test_cov_ts = pos_test_cov_ts + neg_test_cov_ts

        if shuffle:
            train_label_ts, train_cov_ts = self.shuffle(train_label_ts, train_cov_ts)
            test_label_ts, test_cov_ts = self.shuffle(test_label_ts, test_cov_ts)

        return train_label_ts, train_cov_ts, test_label_ts, test_cov_ts

    def convert_df_data_to_time_series(self, df: pd.DataFrame, scaler: Scaler) -> (TimeSeries, TimeSeries):
        """
        convert the data frame to two time series, one for the label and one for the covariance.
        :param df: the data frame to convert to time series.
        :param scaler: the scaler to use.
        :return: label time series, covariant time series
        """
        processed_df = self.preprocess_df(df)
        label_df = processed_df[LABEL_COL + [TIME_COL]]
        covariances_df = processed_df[DYNAMIC_COVARIANT_COLS + STATIC_COVARIANT_COLS + [TIME_COL]]

        label_series = TimeSeries.from_dataframe(df=label_df, time_col=TIME_COL, value_cols=LABEL_COL)
        covariances_series = TimeSeries.from_dataframe(df=covariances_df, time_col=TIME_COL)

        label_series = self.preprocess_time_series(label_series, scaler)
        covariances_series = self.preprocess_time_series(covariances_series, scaler)

        return label_series, covariances_series

    def load_data_as_time_series(self, path_to_data: str = "./Data", scaler: Scaler = None,
                                 max_amount_of_ts_to_load: int = 0) -> (list, list, list, list):
        """
        load the data as time series.
        :param scaler: the scaler to use.
        :param path_to_data: path to the data folder.
        :param max_amount_of_ts_to_load: the maximum amount of time series to load.
        :return: positive and negative time series.

        """
        files_1 = [i for i in glob.glob(f'{path_to_data}/training/*')]
        files_2 = [i for i in glob.glob(f'{path_to_data}/training_setB/*')]
        if max_amount_of_ts_to_load > 0:
            file_list = files_1[:max_amount_of_ts_to_load//2] + files_2[:max_amount_of_ts_to_load//2]
        else:
            file_list = files_1 + files_2
        pos_ts, pos_cov_ts, neg_ts, neg_cov_ts = [], [], [], []
        is_positive = False
        for i, file in enumerate(file_list):
            patient_df = pd.read_csv(file, delimiter='|')
            if patient_df[LABEL_COL].sum().sum() > 0:
                is_positive = True
            label_series, covariances_series = self.convert_df_data_to_time_series(patient_df, scaler)
            if is_positive:
                pos_ts.append(label_series)
                pos_cov_ts.append(covariances_series)
            else:
                neg_ts.append(label_series)
                neg_cov_ts.append(covariances_series)
        return pos_ts, pos_cov_ts, neg_ts, neg_cov_ts

    def preprocess_time_series(self, time_series: TimeSeries, scaler: Scaler) -> TimeSeries:
        """
        preprocess the time series.
        :param time_series: the time series to preprocess.
        :param scaler: the scaler to use.
        :return: the preprocessed time series.
        """
        # Scale the time series
        if scaler is not None:
            time_series = scaler.fit_transform(time_series)

        return time_series

    def preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        preprocess the data frame.
        :param df: the data frame to preprocess.
        :return: the preprocessed data frame.
        """
        # Use fill forward imputation to fill in missing values where possible
        df = df.fillna(method='ffill')
        # Fill other nulls with 0 as model cannot handle nulls
        df = df.fillna(0)

        return df

    def save_split_data(self, train_label_ts: list[TimeSeries], train_cov_ts: list[TimeSeries],
                        test_label_ts: list[TimeSeries], test_cov_ts: list[TimeSeries], base_path: str = "./Data",
                        prefix_name: str = ""):
        """
        save the split data.
        :param train_label_ts: list of darts time series representing the train labels.
        :param train_cov_ts: list of darts time series representing the train covariances.
        :param test_label_ts: list of darts time series representing the test labels.
        :param test_cov_ts: list of darts time series representing the test covariances.
        :param base_path: the base path to save the data.
        :param prefix_name: the prefix name to add to the saved files.
        :return: None
        """
        # save the train data
        with open(f'{base_path}/train_label_ts_{prefix_name}.pkl', 'wb') as f:
            joblib.dump(train_label_ts, f)
        with open(f'{base_path}/train_cov_ts_{prefix_name}.pkl', 'wb') as f:
            joblib.dump(train_cov_ts, f)
        with open(f'{base_path}/test_label_ts_{prefix_name}.pkl', 'wb') as f:
            joblib.dump(test_label_ts, f)
        with open(f'{base_path}/test_cov_ts_{prefix_name}.pkl', 'wb') as f:
            joblib.dump(test_cov_ts, f)
