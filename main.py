import logging
import warnings
from datetime import datetime
from pathlib import Path

from darts.models import TFTModel

from Handlers.DataLoader import DataLoader

warnings.filterwarnings("ignore")

logging.disable(logging.CRITICAL)

if __name__ == '__main__':
    data_loader = DataLoader()
    train_label_ts, train_cov_ts, test_label_ts, test_cov_ts = data_loader.load_data_for_training()

    base_model_name = f"model_V3_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    base_data_path = Path("./Data/ModelTrainingData/")
    base_data_path.mkdir(parents=True, exist_ok=True)
    data_loader.save_split_data(train_label_ts, train_cov_ts, test_label_ts, test_cov_ts,
                                base_path=str(base_data_path),
                                prefix_name=base_model_name)

    input_chunk_length = 1
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
        # loss_fn=MSELoss(),
        # pl_trainer_kwargs={"accelerator": "gpu", "devices": 1},
        random_state=42,
    )
    my_model.fit(series=train_label_ts, past_covariates=train_cov_ts, future_covariates=train_cov_ts,
                 verbose=True)  # , val_series=test_label_ts, val_past_covariates=test_cov_ts)
    my_model.save(work_dir + f"{base_model_name}.pt")

    my_model.to_cpu()

    my_model.save(work_dir + f"{base_model_name}_CPU.pt")

    # print('a')
