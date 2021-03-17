__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'

import os
import re

import numpy as np
import pandas as pd


class data_loader:

    def __init__(self, project_root_dir=".", input_data_repo="data", generated_data_repo="generated_data"):

        self.project_root_dir = project_root_dir

        self.DATA_PATH = os.path.join(project_root_dir, input_data_repo)

        os.makedirs(self.DATA_PATH, exist_ok=True)

        self.OUT_PATH = os.path.join(self.DATA_PATH, generated_data_repo)

        os.makedirs(self.OUT_PATH, exist_ok=True)

    def load(self, filename='radio_data.xlsx'):

        print('Data loading ...')

        filepath = os.path.join(self.DATA_PATH, filename)

        all_dfs = pd.read_excel(filepath, sheet_name=None)

        df = pd.DataFrame()

        for key in all_dfs.keys():
            all_dfs[key]['CIBLE'] = key

            df = df.append(all_dfs[key], ignore_index=True)

        df = self.data_clean(df)

        df = self.transform_data(df)

        print('Data loading finished')

        return df

    def transform_data(self, data):

        stations_df = data.groupby(['STATION'])

        cpt = -1

        columns_list = []

        stations_grp_df = pd.DataFrame()

        for key, item in stations_df:

            grp_matrix = item.sort_values(by=['ECRAN']).groupby(['CIBLE', 'JOUR', 'VAGUE', 'ECRAN'])['GRP'].aggregate(
                'first').unstack()

            cpt += 1

            if (cpt == 0):
                columns_list = grp_matrix.columns

            if (len(grp_matrix.columns) < 38):
                grp_matrix = grp_matrix.reindex(columns=columns_list)

            grp_matrix.columns = np.arange(5.0, 24, 0.5).astype(str)

            grp_matrix['STATION'] = key

            grp_matrix = grp_matrix.reset_index(level=['CIBLE', 'JOUR', 'VAGUE'])

            stations_grp_df = stations_grp_df.append(grp_matrix, ignore_index=True)

        return stations_grp_df

    def format_days(self, day):

        if (isinstance(day, str)):

            day = day.replace('_', '')

            if (not (re.match('S', day) or re.match('D', day))):
                return 'J'

            return day

    def data_clean(self, df):

        df = df.rename(columns={" (ECRAN)": "ECRAN"})

        df = df.rename(columns={"SUPPORT": "STATION"})

        df = df.rename(columns={"TARIF DU 30": "TARIF"})

        df = df.rename(columns={"JOURS": "JOUR"})

        df['JOUR'] = df['JOUR'].apply(lambda jour: self.format_days(jour))

        df.drop(
            ['NB.J.', 'TD', 'CL. PUI.', 'CL. ECO.', 'IND. AFF.', 'CL. AFF.', 'CL. MIX', 'CONTACTS CORRIGES', 'CT/GRP'],
            axis=1, inplace=True)

        df = df[df['STATION'] != 'BFM']

        df = df[df['VAGUE'] != 'Cumul']

        return df
