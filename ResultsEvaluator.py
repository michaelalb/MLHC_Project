import os
from pathlib import Path
from typing import Union, Tuple
import seaborn as sns
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.explainability import tft_explainer
from darts.models import TFTModel
from sklearn.calibration import calibration_curve
import matplotlib.image as mpimg
from PIL import Image


def evaluate_results(results_file_path: Union[str, Path], results_output_path_base: str,
                     train_test_pkl_path: str, model_name: str, results_df_read_path: str = None) -> None:
    """
    A function that evaluates the results of a model
    :param results_file_path: Path to a results pickle file - assumes structure created my ModelEvaluator functions
    :param results_output_path_base: Path to save results for analysis.
    :param results_df_read_path: If not None results will be read from here.
    :param train_test_pkl_path: Path to the train test pickle file
    :param model_name: The name of the model
    :return: None
    """
    if results_df_read_path is not None:
        results_df = pd.read_csv(results_df_read_path)
    else:
        results = joblib.load(results_file_path)
        rows = [
            {**v, 'ts_idx': ts_idx, 'timestamp_reversed': sub_index}
            for ts_idx, sub_dict in results.items()
            for sub_index, v in sub_dict.items()
        ]

        # Create a Pandas DataFrame from the list of rows and reverse the timestamps as they come reversed
        # from predictions
        results_df = pd.DataFrame(rows)
        results_df['max_sub_index'] = results_df.groupby('ts_idx')['timestamp_reversed'].transform('max')
        results_df['timestamp'] = results_df['max_sub_index'] - results_df['timestamp_reversed']
        results_df.drop(columns=['max_sub_index'], inplace=True)
        results_df.sort_values(by=['ts_idx', 'timestamp'], inplace=True)
        results_df.to_csv(results_output_path_base + ".csv", index=False)

    # plot calibration curve
    plot_calibration_curve(results_df, results_output_path_base)

    # plot error and bias
    plot_result_errors_and_biases(results_df, results_output_path_base)

    # plot feature importance and attention
    plot_attention(train_test_pkl_path=train_test_pkl_path,
                   model_name=model_name,
                   results_df=results_df,
                   save_path=results_output_path_base)


def plot_result_errors_and_biases(results_df: pd.DataFrame, save_path: str,
                                  should_show_plots: bool = False) -> None:
    """
    A function that plots the errors and biases of the results
    :param results_df: A results df created by evaluate_results
    :param save_path: The path to save the plot to
    :return: None
    """

    results_df.rename(columns={"hours_of_data_with_sepsis": "Hours of data with sepsis",
                               "hours_of_data": "Hours of data"}, inplace=True)

    results_df['prediction_error'] = abs(results_df['pred'] - results_df['label'])
    # Group the data by Age, Gender, hours of data, and hours of sepsis label
    grouped = results_df.groupby(['Age', 'Gender', 'Hours of data', 'Hours of data with sepsis'])
    # Calculate summary statistics within each group
    summary_stats = grouped.agg({
        'prediction_error': ['mean', 'median'],
        'prob_1': 'min'
    }).reset_index()
    # Rename columns for clarity
    summary_stats.columns = ['Age', 'Gender', 'Hours of data', 'Hours of data with sepsis',
                             'Mean_Error', 'Median_Error', 'Worst_Probability']
    # Sort the summary statistics by Mean_Error or any other metric you prefer
    summary_stats = summary_stats.sort_values(by='Mean_Error', ascending=False)

    # Create a new figure
    plt.figure(figsize=(20, 12))

    # Bin Age into 10-year bins
    summary_stats['Age'] = pd.cut(summary_stats['Age'], bins=range(0, 110, 10), right=False,
                                  labels=[f'{i}-{i + 9}' for i in range(0, 100, 10)])

    # Bin Hours of data into 20-hour bins
    summary_stats['Hours of data'] = pd.cut(summary_stats['Hours of data'], bins=range(0, 360, 20), right=False,
                                            labels=[f'{i}-{i + 19}' for i in range(0, 340, 20)])

    # Keep Hours of data with sepsis values of 9 and 10
    summary_stats = summary_stats[summary_stats['Hours of data with sepsis'].isin([9, 10])]

    # Create a violin plot for Mean_Error by Age
    plt.subplot(2, 2, 1)
    sns.violinplot(x='Age', y='Mean_Error', data=summary_stats)
    plt.title('Mean Error by Age')

    # Create a violin plot for Mean_Error by Gender
    plt.subplot(2, 2, 2)
    sns.violinplot(x='Gender', y='Mean_Error', data=summary_stats)
    plt.title('Mean Error by Gender')

    # Create a violin plot for Mean_Error by Hours of data
    plt.subplot(2, 2, 3)
    sns.violinplot(x='Hours of data', y='Mean_Error', data=summary_stats)
    plt.title('Mean Error by Hours of Data')

    # Create a violin plot for Mean_Error by Hours of data with sepsis
    plt.subplot(2, 2, 4)
    sns.violinplot(x='Hours of data with sepsis', y='Mean_Error', data=summary_stats)
    plt.title('Mean Error by Hours of Data with Sepsis')
    plt.tight_layout()
    if should_show_plots:
        plt.show()
    else:
        plt.savefig(save_path+"error_bias.png")


def plot_calibration_curve(results_df: pd.DataFrame, save_path: str, should_show_plots: bool = False) -> None:
    """
    A function that plots a calibration curve for the results
    :param results_df: A results df created by evaluate_results
    :param save_path: The path to save the plot to
    :param should_show_plots: A boolean indicating whether to show the plots or not
    :return: None
    """
    predicted_probabilities = results_df['prob_1']

    # Extract true labels (1 for positive, 0 for negative)
    true_labels = results_df['label']

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(true_labels, predicted_probabilities, n_bins=10000)

    # Create the calibration curve plot
    plt.figure(figsize=(20, 10))
    plt.plot(prob_pred, prob_true, marker='o', linestyle='--', color='blue', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    if should_show_plots:
        plt.show()
    else:
        plt.savefig(save_path+"calibration_curve.png")


def get_avg_importances(model_explainer: tft_explainer.TFTExplainer, test_label_ts: list[TimeSeries],
                        test_cov_ts: list[TimeSeries], ts_idxs: list[int]) -> Tuple[dict, dict, dict]:
    """
    A function that calculates the average importances of the features and attention heads
    :param model_explainer: A TFTExplainer object
    :param test_label_ts: A list of TimeSeries objects containing the labels for the test set
    :param test_cov_ts: A list of TimeSeries objects containing the covariates for the test set
    :param ts_idxs: A list of indices of the time series to explain
    :return: A dictionary containing the average importances of the features and attention heads
    """

    all_explanations = []
    for ts_idx in ts_idxs:
        positive_ts_label, positive_ts_cov_past, positive_ts_cov_future = [], [], []
        tmp_test_label = test_label_ts[ts_idx]
        tmp_test_cov = test_cov_ts[ts_idx]
        labels_ts = [tmp_test_label[:-timestamp] for timestamp in range(1, len(tmp_test_label))]
        past_covs = [tmp_test_cov[:-timestamp] for timestamp in range(1, len(tmp_test_label))]
        fut_covs = [tmp_test_cov[-(timestamp + 1):] for timestamp in range(1, len(tmp_test_label))]
        for i in range(len(labels_ts)):
            positive_ts_label.append(labels_ts[i])
            positive_ts_cov_past.append(past_covs[i])
            positive_ts_cov_future.append(fut_covs[i])
        try:
            explanations = model_explainer.explain(positive_ts_label, positive_ts_cov_past, positive_ts_cov_future)
            all_explanations.append(explanations)
        except Exception as e:
            print("exception in ts idx: ", ts_idx)
            continue
    encoder_importances = []
    decoder_importances = []
    attention_weights = []
    for explanation in all_explanations:
        for encoder_imp in explanation.get_encoder_importance():
            encoder_importances.append(encoder_imp)
        for decoder_imp in explanation.get_decoder_importance():
            decoder_importances.append(decoder_imp)
        for attn_weights in explanation.get_attention():
            attention_weights.append(attn_weights.values().flatten())
    decoder_imp = pd.concat(decoder_importances)
    encoder_imp = pd.concat(encoder_importances)
    attention_weights = pd.DataFrame(attention_weights)

    return decoder_imp.mean().to_dict(), encoder_imp.mean().to_dict(), \
           attention_weights.mean().to_dict()


def plot_attention(train_test_pkl_path: str, model_name: str, results_df: pd.DataFrame, save_path: str,
                   should_show_plots: bool = False, load_existing_plots: bool = False) -> None:
    """
    A function that shows the attention and feature importance of the model
    :param train_test_pkl_path: The path to the train test pickle file
    :param results_df: A results df created by evaluate_results
    :param save_path: The path to save the plot to
    :param should_show_plots: A boolean indicating whether to show the plots or not
    :param load_existing_plots: A boolean indicating whether to load existing plots or not -
    Should use this if you don't want to wait very long - these plots take a while to create.
    :return: None
    """
    if load_existing_plots:
        for title in ["Sepsis Patient Encoder Feature Importance", "Sepsis Patient Decoder Feature Importance",
                      "Non-Sepsis Patient Encoder Feature Importance", "Non-Sepsis Patient Decoder Feature Importance",
                      "Sepsis Patient Attention", "Non-Sepsis Patient Attention"]:
            image = Image.open(f"{save_path}/{title}.png")
            image.show()
        return

    model_path = f"./Models/{model_name}"
    loaded_model = TFTModel.load(model_path, map_location='cpu')
    loaded_model.load_weights(model_path, map_location='cpu')
    data = joblib.load(train_test_pkl_path)
    train_label_ts, train_cov_ts, test_label_ts, test_cov_ts = data

    # Create explainer object
    # need to pass a background series - so chose arbitrary one
    explainer = tft_explainer.TFTExplainer(loaded_model, background_series=test_label_ts[0][:-2],
                                           background_past_covariates=test_cov_ts[0][:-2],
                                           background_future_covariates=test_cov_ts[0][-3:])

    # Extract feature importances for positive and negative examples
    pos_ts_idx = results_df[results_df['label'] == 1]['ts_idx'].unique()
    positive_examples = results_df[results_df['ts_idx'].isin(pos_ts_idx)]
    negative_examples = results_df[~results_df['ts_idx'].isin(pos_ts_idx)]

    positive_decoder_imp, positive_encoder_imp, positive_attention_weights = get_avg_importances(explainer,
                                                                                                 test_label_ts,
                                                                                                 test_cov_ts,
                                                                                                 pos_ts_idx)
    negative_decoder_imp, negative_encoder_imp, negative_attention_weights = get_avg_importances(explainer,
                                                                                                    test_label_ts,
                                                                                                    test_cov_ts,
                                                                                                    negative_examples[
                                                                                                        'ts_idx'].unique())
    plot_feature_importance(positive_decoder_imp, "Sepsis Patient Encoder Feature Importance",
                            should_show_plots, save_path)
    plot_feature_importance(negative_decoder_imp, "Non-Sepsis Patient Encoder Feature Importance",
                            should_show_plots, save_path)
    plot_feature_importance(positive_encoder_imp, "Sepsis Patient Decoder Feature Importance",
                            should_show_plots, save_path)
    plot_feature_importance(negative_encoder_imp, "Non-Sepsis Patient Decoder Feature Importance",
                            should_show_plots, save_path)
    plot_attention_heat_map(positive_attention_weights, list(positive_decoder_imp.keys()),
                            "Sepsis Patient Attention", should_show_plots, save_path)
    plot_attention_heat_map(negative_attention_weights, list(negative_decoder_imp.keys()),
                            "Non-Sepsis Patient Attention", should_show_plots, save_path)


def plot_attention_heat_map(attention_weights: dict, feature_names:list[str], title: str, should_save: bool, save_path: str):
    """
    A function that plots the attention weights of the model
    :param attention_weights: dict with average attention weights of each feature
    :param title: title of the plot
    :param should_save: whether to save the plot
    :param save_path: path to save the plot
    :return: None
    """
    attention_weights_df = pd.DataFrame(attention_weights, columns=feature_names)
    plt.figure(figsize=(20, 10))
    sns.heatmap(attention_weights_df, annot=True, cmap="YlGnBu")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.title(title)
    if should_save:
        plt.savefig(f"{save_path}/{title}.png")
    else:
        plt.show()


def plot_feature_importance(importance_dict: dict, title: str, should_save: bool,
                            save_path: str):
    """
    A function that plots the feature importance of the model
    :param importance_dict: dict with average importance of each feature
    :param title: title of the plot
    :param should_save: whether to save the plot
    :param save_path: path to save the plot
    :return: None
    """
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features, values = zip(*sorted_importance)
    y_labels = [i.strip('_futcov').strip('_pastcov') for i in features]

    plt.figure(figsize=(20, 10))
    plt.bar(y_labels, values)
    plt.xlabel("Feature Importance")
    plt.xticks(rotation=45)
    plt.ylabel("Features")
    plt.title(title)
    if should_save:
        plt.savefig(f"{save_path}/{title}.png")
    else:
        plt.show()


if __name__ == '__main__':
    model_name = "model_V3_CPU_2023_08_27_15_24_22.pt"
    evaluate_results(results_file_path=f"./ResultsEvaluation/{model_name}_ts_train_test_results.pkl",
                     results_output_path_base=f"./ResultsEvaluation/{model_name}_ts_train_test_results",
                     train_test_pkl_path="./Data/ffill_ts_train_test.pkl",
                     results_df_read_path=f"./ResultsEvaluation/{model_name}_ts_train_test_results.csv",
                     model_name=model_name)