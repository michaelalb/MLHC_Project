import logging
import warnings
from datetime import datetime
from pathlib import Path

from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression

from DataLoader import DataLoader

warnings.filterwarnings("ignore")

logging.disable(logging.CRITICAL)
quantiles_sparse = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
quantiles_full = [num / 100.0 for num in range(1, 100)]


def run_training_loop(model_version=7):
    """
    Runs the model training loop, assumes the project file structure.
    :param model_version: The version of the model to train - indidcation which iteration of model training this is
    :return:
    """

    # load and create the data for training
    data_loader = DataLoader()
    train_label_ts, train_cov_ts, test_label_ts, test_cov_ts = data_loader.load_data_for_training()
    base_model_name = f"model_V{model_version}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    base_data_path = Path("./Data/ModelTrainingData/")
    base_data_path.mkdir(parents=True, exist_ok=True)
    data_loader.save_split_data(train_label_ts, train_cov_ts, test_label_ts, test_cov_ts,
                                base_path=str(base_data_path),
                                prefix_name=base_model_name)

    # define the model
    input_chunk_length = 10
    forecast_horizon = 1
    work_dir = './Models/'
    my_model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=16,
        n_epochs=1,
        work_dir=work_dir,
        add_relative_index=False,
        add_encoders=None,
        save_checkpoints=True,
        likelihood=QuantileRegression(quantiles=quantiles_full),
        # pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}, # Used on cluster
        random_state=42,
    )
    # train the model
    my_model.fit(series=train_label_ts, past_covariates=train_cov_ts, future_covariates=train_cov_ts, verbose=True)

    # save two versions of the model - assumes training happens on a GPU
    my_model.save(work_dir + f"{base_model_name}.pt")
    my_model.to_cpu()
    my_model.save(work_dir + f"{base_model_name}_CPU.pt")


if __name__ == '__main__':
    run_training_loop()

