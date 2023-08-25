import glob

import tsdb
import pandas as pd
import pandas as pd
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.models import TFTModel

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.transformers import MissingValuesFiller
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)

def preprocess_time_series(time_series):
    # Fill missing values if necessary
    time_series = MissingValuesFiller().transform(time_series)

    # Scale the time series
    transformer = Scaler()
    time_series = transformer.fit_transform(time_series)

    return time_series


if __name__ == '__main__':
    files_1 = [i for i in glob.glob('./Data/training/*')]
    files_2 = [i for i in glob.glob('./Data/training_setB/*')]
    time_series, covs = [], []
    for f in files_1:
        patient_df = pd.read_csv(f, delimiter='|')
        label_col = ["SepsisLabel"]
        dynamic_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
           'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
           'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
           'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
           'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
           'Fibrinogen', 'Platelets']
        static_cols = ['Age', 'Gender', 'Unit1', 'Unit2',"HospAdmTime"]
        time_column = "ICULOS"
        label_column = patient_df[label_col + [time_column]]
        covariates_df = patient_df[dynamic_cols + static_cols + [time_column]].fillna(0)
        # dynamic_columns = patient_df[dynamic_cols + [time_column]]  # Replace with actual dynamic column names
        # static_columns = patient_df[static_cols + [time_column]]  # Replace with actual static column names

        # Create a Darts time series for the label column
        label_series = TimeSeries.from_dataframe(df=label_column, time_col=time_column, value_cols=label_col)
        label_series = preprocess_time_series(label_series)
        # If you have a timestamp column, set it as the time index:
        # label_series = label_series.set_time_index('timestamp_column_name')

        # Create time series for dynamic covariates
        covarities = TimeSeries.from_dataframe(covariates_df, time_column)
        covarities = preprocess_time_series(covarities)
        # Combine label, dynamic, and static series into a single time series
        # patient_time_series = label_series.stack(dynamic_series)

        time_series.append(label_series)
        covs.append(covarities)
        if len(time_series) >200:
            break

    # Preprocess your list of time series
    # preprocessed_time_series = [preprocess_time_series(ts) for ts in time_series]

    # Define the prediction length and seasonality (if known)
    prediction_length = 7  # Adjust this as needed
    seasonality = check_seasonality(time_series[0])  # Automatically detect seasonality

    # Instantiate and train the Temporal Fusion Transformer model
    input_chunk_length = 1#seasonality[1]
    forecast_horizon = 1
    dir = './'
    my_model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=16,
        n_epochs=2,
        work_dir='./',
        add_relative_index=False,
        add_encoders=None,
        save_checkpoints=True,
        # loss_fn=MSELoss(),
        random_state=42,
    )
    my_model.fit(time_series[0], future_covariates=covs[0], verbose=True)
    my_model.save(dir + "V0_model.pt")
    print('a')

