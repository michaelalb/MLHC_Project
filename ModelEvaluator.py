import logging
import warnings
from pathlib import Path
from typing import Union

import joblib
from darts import TimeSeries
from darts.models import TFTModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

from OfficialEvaluations import evaluate_sepsis_score

warnings.filterwarnings("ignore")

logging.disable(logging.CRITICAL)


def estimate_probability_from_quantiles(quantiles: list[float], quantile_values: list[float],
                                        value_of_interest: float) -> float:
    """
    Estimate the probability of a value of interest from a list of quantiles and their values
    :param quantiles: the quantiles
    :param quantile_values: the values of the quantiles
    :param value_of_interest: the value of interest - here the label we were predicting
    :return: the probability of the value of interest
    """
    normalized_quantiles = [q / 100.0 for q in quantiles] if quantiles[0] > 1 else quantiles
    if value_of_interest < quantile_values[0]:
        q1 = 0
        q2 = normalized_quantiles[0]
        value1 = 0
        value2 = quantile_values[0]
    if value_of_interest > quantile_values[-1]:
        q1 = normalized_quantiles[-1]
        q2 = 1
        value1 = quantile_values[-1]
        value2 = 1
    # Find the closest quantiles
    for i in range(len(normalized_quantiles) - 1):
        if quantile_values[i] <= value_of_interest <= quantile_values[i + 1]:
            q1 = normalized_quantiles[i]
            q2 = normalized_quantiles[i + 1]
            value1 = quantile_values[i]
            value2 = quantile_values[i + 1]
            break

    # Linear interpolation to estimate cumulative probability
    P = q1 + (q2 - q1) * (value_of_interest - value1) / (value2 - value1)

    # Calculate the probability
    probability = 1 - P
    if probability < 0.5:
        print("Warning: probability is less than 0.5 - for value of interest: ", value_of_interest,
              " and quantiles: ", f"{quantile_values}")
    return probability


def create_results_dict(model: TorchForecastingModel, test_label_ts: list[TimeSeries], test_cov_ts: list[TimeSeries],
                        output_path: str,
                        should_save: bool = True,
                        results_file: str = None) -> dict:
    """
    Create a dictionary of results for each time series in the test set
    :param model: The TFT model to use for prediction
    :param test_label_ts: test label time series (list of sepsis label time series)
    :param test_cov_ts: test covariates  time series (list of feature time series)
    :param output_path: path to save the results
    :param should_save: whether to save the results
    :param results_file: if not None - results will be loaded from there instead of being calculated.
    :return: dictionary of results in the format: {patient index:
    {prediction time stamp: {'pred': the prediction made, 'label': label, 'prob_0': probability for zero,
    'prob_1': probability for one}, ...}, ...}
    """
    if results_file is not None:
        results = joblib.load(results_file)
        return results
    results = {}
    for ts_idx in range(len(test_label_ts)):
        tmp_test_label = test_label_ts[ts_idx]
        tmp_test_cov = test_cov_ts[ts_idx]
        # predict
        labels_ts = [tmp_test_label[:-timestamp] for timestamp in range(1, len(tmp_test_label))]
        past_covs = [tmp_test_cov[:-timestamp] for timestamp in range(1, len(tmp_test_label))]
        fut_covs = [tmp_test_cov[-(timestamp + 1):] for timestamp in range(1, len(tmp_test_label))]
        true_labels = [tmp_test_label[-timestamp] for timestamp in range(1, len(tmp_test_label))]
        preds = model.predict(1, labels_ts, past_covariates=past_covs, future_covariates=fut_covs,
                              verbose=False)
        preds_proba = model.predict(1, labels_ts, past_covariates=past_covs, future_covariates=fut_covs,
                                    verbose=False, predict_likelihood_parameters=True)
        probs = []
        for i, pred_proba in enumerate(preds_proba):
            quantiles = [float(q.replace("SepsisLabel_q", "")) for q in pred_proba.components]
            quantile_values = list(pred_proba.values()[0])
            label = true_labels[i].values()[0][0]
            probability = estimate_probability_from_quantiles(quantiles, quantile_values, label)
            prob = (probability, 1 - probability) if label == 0 else (1 - probability, probability)
            probs.append(prob)

        results[ts_idx] = {len(probs) - i: {'pred': round(pred.values()[0][0], 1), 'label': label.values()[0][0],
                                            'prob_0': prob[0], 'prob_1': prob[1], "ts_idx": ts_idx,
                                            "Age": tmp_test_cov['Age'].values()[0][0],
                                            "Gender": tmp_test_cov['Gender'].values()[0][0], "hours_of_data": len(tmp_test_label),
                                            "hours_of_data_with_sepsis": len(tmp_test_label.values()[tmp_test_label.values() == 1])}
                           for i, (pred, label, prob) in enumerate(zip(preds, true_labels, probs))}
    if should_save:
        joblib.dump(results, output_path)

    return results


def create_official_eval_results_file(
        results: dict,
        base_label_path: Union[str, Path],
        base_pred_path: Union[str, Path]) -> None:
    """
    Create the official evaluation results file which expects files in a very specific format
    :param results: results dict to transform - expects the format create by the method create_results_dict
    :param base_label_path: path to save the label file
    :param base_pred_path: path to save the prediction file
    :return: None
    """
    if isinstance(base_label_path, str):
        base_label_path = Path(base_label_path)
    if isinstance(base_pred_path, str):
        base_pred_path = Path(base_pred_path)
    base_label_path.mkdir(parents=True, exist_ok=True)
    base_pred_path.mkdir(parents=True, exist_ok=True)

    for patient_id, patient_results in results.items():
        label_path = base_label_path / f"{patient_id}.psv"
        pred_path = base_pred_path / f"{patient_id}.psv"

        with open(label_path, "w") as label_file, open(pred_path, "w") as pred_file:
            label_file.write("timestamp|SepsisLabel\n")
            pred_file.write(f"PredictedLabel|PredictedProbability\n")
            for timestamp, timestamp_results in  sorted(patient_results.items(), key=lambda x: x[0], reverse=True):
                label_file.write(f"{timestamp}|{timestamp_results['label']}\n")
                pred_file.write(f"{timestamp_results['pred']}|{timestamp_results['prob_1']}\n")


def evaluate_model(model_name: str, train_test_data_pickle_file_name: str,
                   evaluation_results_output_path: str, results_file: str = None) -> dict:
    """
    Orchestrator function for evaluating a model
    :param model_name: name of the model to be evaluated, must sit in a "Models" folder with checkpoint according to
    darts format
    :param train_test_data_pickle_file_name: the train\test data pickle file name, must sit in a "Data" folder,
    to avoid evaluating on the same data the model was trained on.
    :param evaluation_results_output_path: path to save the evaluation results.
    :param results_file: if not None - results will be loaded from there instead of being calculated.
    :return: dictionary of evaluation results - auroc, auprc, accuracy, f_measure, utility function
    """
    # Load the data and the model and evaluate on the train set
    model_path = f"./Models/{model_name}"
    loaded_model = TFTModel.load(model_path, map_location='cpu')
    loaded_model.load_weights(model_path, map_location='cpu')
    data = joblib.load(f"./Data/{train_test_data_pickle_file_name}")
    train_label_ts, train_cov_ts, test_label_ts, test_cov_ts = data
    results = create_results_dict(loaded_model, test_label_ts, test_cov_ts, evaluation_results_output_path, True,
                                  results_file)

    # Create the official evaluation results file
    base_label_path: str = "OfficialEvaluationInput/label_directory"
    base_pred_path: str = "OfficialEvaluationInput/prediction_directory"
    create_official_eval_results_file(results, base_label_path, base_pred_path)

    # Evaluate the results using the official Physionet challenge evaluation script
    auroc, auprc, accuracy, f_measure, normalized_observed_utility = evaluate_sepsis_score(base_label_path,
                                                                                           base_pred_path)

    # Print the results
    print(f"AUROC: {auroc}")
    print(f"AUPRC: {auprc}")
    print(f"Accuracy: {accuracy}")
    print(f"F-measure: {f_measure}")
    print(f"NUO: {normalized_observed_utility}")

    return {'AUROC': auroc, 'AUPRC': auprc, 'Accuracy': accuracy, 'F-measure': f_measure, 'NUO': normalized_observed_utility}


if __name__ == '__main__':
    model_name = 'model_V3_CPU_2023_08_27_15_24_22.pt'
    evaluate_model(model_name=model_name,
                   train_test_data_pickle_file_name="ffill_ts_train_test.pkl",
                   evaluation_results_output_path=f"./ResultsEvaluation/{model_name}_ts_train_test_results.pkl")#,
                   # results_file="./ffill_ts_train_test_results.pkl")
