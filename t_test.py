# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:24:15 2022

@author: Mert
"""
import numpy as np
from collections import defaultdict
import os
from os import listdir
import glob

import pandas as pd

path_to_dir = './fold_results/'


def get_fold_statistic(path_to_dir):
    
    path_to_dir_runs = [x[0] for x in os.walk(path_to_dir)][1:]
    fold_statistics = defaultdict(lambda: defaultdict(dict))
    for path_to_dir_run in path_to_dir_runs:
        
        for filename in [
            filename for filename in listdir(
                path_to_dir_run
                ) if filename.endswith('csv')
            ]:
            run_name = path_to_dir_run.split('/')[-1]
            if '(' in filename:
                data_name = filename.split('_')[0]
                model_name = filename.split('_')[1]
            else:
                data_name = filename.split(
                    'fold_results_'
                    )[-1].split('_')[0]
                model_name = filename.split(
                    'fold_results_'
                    )[-1].split('_')[-1].split('.csv')[0]
            fold_statistics[
                run_name
                ][
                    data_name
                    ][
                        model_name
                        ] = pd.read_csv(
                            path_to_dir_run + '/'+ filename, index_col=0
                            )
    return fold_statistics

model_to_num = {
    'dsm':0 , 
    'dcm':1,
    'cph':2, 
    'deephit':3,
    'dha':4, 
    'survivalforest':5, 
    'time':6
    }

train_test_size = {
    'flchain':{'train_size':4566, 'test_size':1305},
    'support':{'train_size':6211, 'test_size':1775},
    'metabric':{'train_size':1332, 'test_size':381},
    'pbc':{'train_size':1361, 'test_size':389},
    }


fold_statistics = get_fold_statistic(path_to_dir)

def get_score_diff(fold_statistics):

    score_diff = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    run_size = len(fold_statistics.keys())
    
    for run_id, run_name in enumerate(fold_statistics.keys()):
        for data_name in fold_statistics[run_name].keys():
            for model_name in fold_statistics[run_name][data_name].keys():
                model_size = len(fold_statistics[run_name][data_name].keys())
                model_stats = fold_statistics[run_name][data_name][model_name]
                for stat_name in model_stats.index:
                    fold_size = model_stats.shape[1]
                    score_diff[data_name][stat_name] = np.zeros(
                        (run_size, fold_size, model_size,model_size)
                        ) 
    
    for run_id, run_name in enumerate(fold_statistics.keys()):
        for data_name in fold_statistics[run_name].keys():
            for model1_name in fold_statistics[run_name][data_name].keys():      
                for model2_name in fold_statistics[run_name][data_name].keys():
                    id1 = model_to_num[model1_name]
                    id2 = model_to_num[model2_name]
                    model1_stats = fold_statistics[run_name][data_name][model1_name]
                    model2_stats = fold_statistics[run_name][data_name][model2_name]
                    for i, model1_row in enumerate(model1_stats.iterrows()):
                        model1_row = model1_row[1]
                        model1_stats = model1_row.name
                        try:
                            model2_row = model2_stats.iloc[i]
                            for fold_name,(r1, r2) in enumerate(
                                    zip(model1_row, model2_row)
                                    ):
                                if model1_name.lower() == 'dha':
                                    r1 = float(r1)
                                if model2_name.lower() == 'dha':
                                    r2 = float(r2)
                                if not isinstance(r1,str):
                                    score_diff[
                                        data_name
                                        ][
                                            model1_stats
                                            ][
                                                run_id
                                                ][
                                                    fold_name
                                                    ][id1][id2] = r1 - r2
                        except:
                            pass
                            
    return score_diff

score_diff = get_score_diff(fold_statistics)

mean_diff = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

for data_name in score_diff.keys():
    for stats_name in score_diff[data_name].keys():
        experiment_size = score_diff[data_name][stats_name].shape[0]
        experiment_size *= score_diff[data_name][stats_name].shape[1]
        mean_diff[
            data_name
            ][
                stats_name
                ] =  score_diff[data_name][
                    stats_name
                    ].sum(0).sum(0) / experiment_size

std_diff = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))


for data_name in score_diff.keys():
    for stats_name in score_diff[data_name].keys():
        experiment_size = score_diff[data_name][stats_name].shape[0]
        experiment_size *= score_diff[data_name][stats_name].shape[1]
        ratio = train_test_size[
            data_name
            ]['test_size'] / train_test_size[
                data_name
                ]['train_size']

        std_diff[data_name][stats_name] = score_diff[data_name][stats_name]
        std_diff[data_name][stats_name] -= np.expand_dims(
            np.expand_dims(mean_diff[data_name][stats_name], 0), 0
            )
        std_diff[data_name][stats_name] = std_diff[data_name][stats_name]**2
        std_diff[data_name][stats_name] = std_diff[data_name][stats_name].sum(0).sum(0)
        std_diff[data_name][stats_name] = std_diff[data_name][stats_name] / (experiment_size-1)
        std_diff[data_name][stats_name] *= (ratio + 1/experiment_size)
        std_diff[data_name][stats_name] = np.sqrt(
            std_diff[data_name][stats_name]
            )

                    
                    
            
            
t_value_diff = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

for data_name in score_diff.keys():
    for stats_name in score_diff[data_name].keys():
        t_value_diff[
            data_name
            ][
                stats_name
                ] = mean_diff[
            data_name
            ][
                stats_name
                ] / std_diff[
            data_name
            ][
                stats_name
                ]

        
