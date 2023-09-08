
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class DataExplorer:
    def __init__(self, full_df_path: Union[str, Path]):
        self.full_df = pd.read_parquet(str(full_df_path))
        self.feature_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
                             'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
                             'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
                             'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
                             'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
                             'Fibrinogen', 'Platelets']

    def plot_feature_distributions_by_age_gender(self, save_path: Union[str,Path]):
        """
        create plots to observe age and gender biases in data distributions
        :param save_path: The path to save the plots to
        :return: None
        """
        if isinstance(save_path, str):
            save_path = Path(save_path)

        save_path.mkdir(parents=True, exist_ok=True)
        # Create age groups in 10-year intervals up to a maximum of 100
        age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        age_labels = [f"{start}-{end}" for start, end in zip(age_bins[:-1], age_bins[1:])]

        # Add a new column 'AgeGroup' to your DataFrame with age groups
        self.full_df['AgeGroup'] = pd.cut(self.full_df['Age'], bins=age_bins, labels=age_labels)

        # Loop through each numeric column and create a violin plot for age and gender groups
        for column in self.feature_cols:
            plt.figure(figsize=(20, 10))
            sns.violinplot(x='AgeGroup', y=column, hue='Gender', data=self.full_df.dropna(subset=[column]), split=True)
            plt.title(f'Violin Plot of {column} by Age and Gender')
            plt.xlabel('Age Group')
            plt.ylabel(column)
            plt.xticks(rotation=45)  # Rotate x-axis labels for readability
            plt.legend(title='Gender', labels=['Female (0)', 'Male (1)'])  # Customize legend labels
            plt.savefig(str(save_path / f'{column}.jpeg'))
            plt.close()  # Close the current plot to release resources

        # Remove the 'AgeGroup' column to avoid altering the original DataFrame
        self.full_df.drop('AgeGroup', axis=1, inplace=True)

    def plot_feature_distributions(self, save_path: Union[str, Path], type: str = 'kde'):
        """
        Plot the distribution of each feature in the dataset
        :param save_path: The path to save the plots to
        :param type: The type of plot to use. Options are 'kde' and 'hist'
        :return: None
        """
        if isinstance(save_path, str):
            save_path = Path(save_path)
        if str(save_path).find(type) == -1:
            save_path_type = save_path / type
        else:
            save_path_type = save_path
        save_path_type.mkdir(parents=True, exist_ok=True)
        # Iterate through columns and create distribution plots
        for i, column_name in enumerate(self.feature_cols):
            # Plot the distribution as a KDE curve
            plt.figure(figsize=(20, 10))
            self.full_df[[column_name, 'id']].dropna().groupby('id').mean().plot(kind=type, color='blue')
            plt.tight_layout()
            plt.title(f'{type} plot for feature -  {column_name}')
            plt.xlabel('Average value per patient - disregarding nulls')
            if type == 'kde':
                plt.ylabel('Density')
            else:
                plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(str(save_path_type / f'{column_name}.jpeg'))

    def plot_feature_counts(self, save_path: Union[str, Path]):
        """
        Plot the count of each feature in the dataset
        :param save_path: The path to save the plots to
        :return: None
        """
        if isinstance(save_path, str):
            save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        # Iterate through columns and create distribution plots
        avgs = []
        for i, column_name in enumerate(self.feature_cols):
            # Plot the distribution as a KDE curve
            percent_of_non_null_vals = self.full_df[[column_name, 'id']].dropna().groupby('id')[column_name].size() / self.full_df[[column_name, 'id']].groupby('id')[column_name].size()
            avg_percent_of_non_null_vals = percent_of_non_null_vals.mean()
            avgs.append((avg_percent_of_non_null_vals, column_name))
        avgs.sort()
        vals = [i[0] for i in avgs]
        names = [i[1] for i in avgs]
        plt.figure(figsize=(20, 10))
        plt.bar(names, vals)
        plt.xticks(rotation=45, fontsize=6)

        plt.title(f'Average percent of non null values across features')
        plt.xlabel('Index of feature')
        plt.ylabel('Average percent of non null values')
        plt.savefig(str(save_path / 'Non-null values per feature.jpeg'))

    def plot_patient_gender_count_by_age(self, save_path: Union[str, Path]):
        """
        Create plots to observe the count of male and female patients by age groups.

        :param save_path: The path to save the plots to.
        :return: None
        """
        if isinstance(save_path, str):
            save_path = Path(save_path)

        save_path.mkdir(parents=True, exist_ok=True)

        # Create age groups in 10-year intervals up to a maximum of 100
        age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        age_labels = [f"{start}-{end}" for start, end in zip(age_bins[:-1], age_bins[1:])]

        # Add a new column 'AgeGroup' to your DataFrame with age groups
        self.full_df['AgeGroup'] = pd.cut(self.full_df['Age'], bins=age_bins, labels=age_labels)

        # Group the data by age group, 'file_name' (one row per patient), and 'Gender', then count the occurrences
        gender_count = self.full_df.groupby(['AgeGroup', 'file_name', 'Gender']).size().unstack(fill_value=0)

        # Calculate the total count of male and female patients in each age group
        gender_count['Total'] = gender_count.sum(axis=1)

        # Reset the index to create a DataFrame suitable for plotting
        gender_count = gender_count.reset_index()

        # Melt the DataFrame to reshape it for plotting
        gender_count_melted = gender_count.melt(id_vars=['AgeGroup'], value_vars=[0, 1, 'Total'],
                                                var_name='Gender', value_name='Count')

        # Create a bar plot for male, female, and total patient counts by age group
        plt.figure(figsize=(20, 10))
        sns.barplot(x='AgeGroup', y='Count', hue='Gender', data=gender_count_melted)
        plt.title('Patient Gender Count by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.xticks(rotation=45)  # Rotate x-axis labels for readability
        plt.legend(title='Gender', labels=['Female', 'Male', 'Total'])
        plt.savefig(str(save_path / 'patient_gender_count_by_age.jpeg'))
        plt.close()  # Close the current plot to release resources

        # Remove the 'AgeGroup' column to avoid altering the original DataFrame
        self.full_df.drop('AgeGroup', axis=1, inplace=True)


if __name__ == '__main__':
    data_explorer = DataExplorer(Path('Data/full_data.parquet'))
    data_explorer.plot_feature_counts('../Figures/')
    data_explorer.plot_patient_gender_count_by_age('../Figures/')
    data_explorer.plot_feature_distributions_by_age_gender('../Figures/AgeGenderDists')
    data_explorer.plot_feature_distributions('../Figures/FeatureDists')
    data_explorer.plot_feature_distributions('../Figures/FeatureDists', type='hist')
