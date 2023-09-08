from pathlib import Path
import pandas as pd
import numpy as np
from darts.models import TFTModel
import joblib
import logging
import warnings
warnings.filterwarnings("ignore")

logging.disable(logging.CRITICAL)


def estimate_probability_from_quantiles(quantiles: list[float], quantile_values: list[float],
                                        value_of_interest: float) -> float:

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

    return probability


if __name__ == '__main__':
    model_path = r'./Models/model_V3_CPU_2023_08_27_15_24_22.pt'
    loaded_model = TFTModel.load(model_path, map_location='cpu')
    loaded_model.load_weights(model_path, map_location='cpu')

    data = joblib.load("Data/ffill_ts_train_test.pkl")

    train_label_ts, train_cov_ts, test_label_ts, test_cov_ts = data

    results = {}
    for ts_idx in range(len(test_label_ts)):
        tmp_test_label = test_label_ts[ts_idx]
        tmp_test_cov = test_cov_ts[ts_idx]
        # predict
        labels_ts = [tmp_test_label[:-timestamp] for timestamp in range(1, len(tmp_test_label))]
        past_covs = [tmp_test_cov[:-timestamp] for timestamp in range(1, len(tmp_test_label))]
        fut_covs = [tmp_test_cov[-(timestamp + 1):] for timestamp in range(1, len(tmp_test_label))]
        true_labels = [tmp_test_label[-timestamp] for timestamp in range(1, len(tmp_test_label))]
        preds = loaded_model.predict(1, labels_ts, past_covariates=past_covs, future_covariates=fut_covs,
                                     verbose=False)
        preds_proba = loaded_model.predict(1, labels_ts, past_covariates=past_covs, future_covariates=fut_covs,
                                     verbose=False, predict_likelihood_parameters=True)
        probs = []
        for i, pred_proba in enumerate(preds_proba):
            quantiles = [float(q.replace("SepsisLabel_q", "")) for q in pred_proba.components]
            quantile_values = list(pred_proba.values()[0])
            value_of_interest = preds[i].values()[0][0]
            label = true_labels[i].values()[0][0]
            probability = estimate_probability_from_quantiles(quantiles, quantile_values, value_of_interest)
            prob = (probability, 1-probability) if label == 0 else (1-probability, probability)
            probs.append(prob)

        results[ts_idx] = {i: {'pred': round(pred.values()[0][0], 1), 'label': label.values()[0][0], 'prob_0': prob[0],
                               'prob_1': prob[1]}
                           for i, (pred, label, prob) in enumerate(zip(preds, true_labels, probs))}

    joblib.dump(results, "./ffill_ts_train_test_results.pkl")

    base_eval_folder_path = Path(r"C:\Users\t-mialbu\PycharmProjects\MLHC_Project\Evaluation")
    base_label_path = base_eval_folder_path / "label_directory"
    base_pred_path = base_eval_folder_path / "prediction_directory"
    base_label_path.mkdir(parents=True, exist_ok=True)
    base_pred_path.mkdir(parents=True, exist_ok=True)

    for patient_id, patient_results in results.items():
        label_path = base_label_path / f"{patient_id}.psv"
        pred_path = base_pred_path / f"{patient_id}.psv"

        # res_df = pd.DataFrame.from_dict(patient_results, orient='index')
        # res_df.rename(columns={'pred': "SepsisLabel", "label": "PredictedLabel", "prob_1": "PredictedProbability"},
        #               inplace=True)
        # np.savetxt(str(label_path), res_df['SepsisLabel'], delimiter='|')
        # np.savetxt(str(pred_path), res_df[['PredictedLabel', 'PredictedProbability']], delimiter='|')
        with open(label_path, "w") as label_file, open(pred_path, "w") as pred_file:
            label_file.write("timestamp|SepsisLabel\n")
            pred_file.write(f"PredictedLabel|PredictedProbability\n")
            for timestamp, timestamp_results in patient_results.items():
                label_file.write(f"{timestamp}|{timestamp_results['label']}\n")
                pred_file.write(f"{timestamp_results['pred']}|{timestamp_results['prob_1']}\n")

    from OfficialEvaluations import evaluate_sepsis_score
    auroc, auprc, accuracy, f_measure, normalized_observed_utility = evaluate_sepsis_score(base_label_path, base_pred_path)


    print(f"AUROC: {auroc}")
    print(f"AUPRC: {auprc}")
    print(f"Accuracy: {accuracy}")
    print(f"F-measure: {f_measure}")
    print(f"NUO: {normalized_observed_utility}")


