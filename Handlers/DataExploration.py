
from pathlib import Path
from typing import Union

import pandas as pd
import matplotlib.pyplot as plt


class DataExplorer:
    def __init__(self, full_df_path: Union[str, Path]):
        self.full_df = pd.read_parquet(str(full_df_path))

    def plot_feature_distributions(self,save_path: Union[str, Path], type: str = 'kde'):
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
        column_names = [i for i in self.full_df.columns if i not in ['id', 'data_set', 'file_name']]
        # Iterate through columns and create distribution plots
        for i, column_name in enumerate(column_names):
            # Plot the distribution as a KDE curve
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
        column_names = [i for i in self.full_df.columns if i not in ['id', 'data_set', 'file_name']]
        # Iterate through columns and create distribution plots
        avgs = []
        for i, column_name in enumerate(column_names):
            # Plot the distribution as a KDE curve
            percent_of_non_null_vals = self.full_df[[column_name, 'id']].dropna().groupby('id')[column_name].size() / self.full_df[[column_name, 'id']].groupby('id')[column_name].size()
            avg_percent_of_non_null_vals = percent_of_non_null_vals.mean()
            avgs.append((avg_percent_of_non_null_vals, column_name))
        avgs.sort()
        vals = [i[0] for i in avgs]
        names = [i[1] for i in avgs]
        plt.bar(names, vals)
        plt.xticks(rotation=45, fontsize=6)

        plt.title(f'Average percent of non null values across features')
        plt.xlabel('Index of feature')
        plt.ylabel('Average percent of non null values')
        plt.savefig(str(save_path / 'Non-null values per feature.jpeg'))


if __name__ == '__main__':
    data_explorer = DataExplorer(Path('../Data/full_data.parquet'))
    data_explorer.plot_feature_counts('../Figures/')
    data_explorer.plot_feature_distributions('../Figures/FeatureDists')
    data_explorer.plot_feature_distributions('../Figures/FeatureDists', type='hist')
    print('a')