# -*- coding: utf-8 -*-
import numpy as np 
import glob
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from pandas import read_csv


if __name__ == '__main__':
    # select the model
    #model_name = "REV" #{"HPGAN", "MVAE", "REV", "TF"}
    model_names = ["HPGAN", "MVAE", "Dance_Rev"]#, "TransFlower"]    
    n_frames = " 200 frames"
    n = 200
    for model_name in model_names:
        
        file_name = [model_name + '_gen', model_name + '_gnd']

        #load feature matrix of two dance clips with 2 windows for comparison.          
        csv_file_a = os.path.join('results','lma_feats','feature_mat_'+ file_name[0]+str(n) + '.csv')
        df_a = read_csv(csv_file_a)

        csv_file_b = os.path.join('results','lma_feats','feature_mat_' + file_name[1] +str(n)+ '.csv')
        df_b = read_csv(csv_file_b)

        correlation_f = []
        #correlation of body features ('ϕ0'-'ϕ31')

        body_f_n = 32
        body_f = ['ϕ'+str(k) for k in range(body_f_n)]
        df_ab = df_a[body_f]
        #print(df_ab)
        df_bb = df_b[body_f]   
        correlation_f.append(df_ab.corrwith(df_bb, axis = 1, method = 'pearson'))

        #tot_c_b = np.sum(c_b)
        #print(c_b[0].round(2))

        #correlation of effort features ('ϕ32'-'ϕ53')
        effort_f_n = 54
        effort_f = ['ϕ'+str(k) for k in range(32, effort_f_n)]
        df_ae = df_a[effort_f]
        #print(df_ae)
        df_be = df_b[effort_f]
        correlation_f.append(df_ae.corrwith(df_be, axis = 1, method = 'pearson'))
        #tot_c_b = np.sum(c_b)
        #print(c_e[0].round(2))

        #correlation of shape features ('ϕ54'-'ϕ82')
        shape_f_n = 59
        shape_f = ['ϕ'+str(k) for k in range(54, shape_f_n)]
        df_as = df_a[shape_f]
        #print(df_as)
        df_bs = df_b[shape_f]
        correlation_f.append(df_as.corrwith(df_bs, axis = 1, method = 'pearson'))
        #tot_c_b = np.sum(c_b)
        #print(c_s[0].round(2))

        #correlation of space features ('ϕ83'-'ϕ84')
        space_f_n = 61
        space_f = ['ϕ'+str(k) for k in range(59, space_f_n)]
        df_ac = df_a[space_f]
        #print(df_ac)
        df_bc = df_b[space_f]
        correlation_f.append(df_ac.corrwith(df_bc, axis = 1, method = 'pearson'))
        #tot_c_b = np.sum(c_b)
        #print(c_c[0].round(2))

        weight = np.array([0.325, 0.325, 0.20, 0.15])
#       weight = np.array([0.5, 0.5])#, 0.2]) # without space and shape

        n_win = len(correlation_f[0])
        tot_cor = np.zeros(n_win)    

        for i,cor in enumerate(correlation_f):
            tot_cor += (cor*100) *weight[i]

            #print(tot_cor)
        barWidth = 0.15
        fig, ax = plt.subplots(figsize =(16, 9))
        #f.set_figwidth(4)
        #f.set_figheight(1)
        #subplots(1, 2, figsize = (25, 12))#subplots(figsize =(12, 8))
        br1 = np.arange(n_win)
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]
        v = [str(k+1) for k in range(n_win)]
        #print(v) 

        ax.bar(br1, correlation_f[0], color ='r', width = barWidth,
            edgecolor ='grey', label ='Body')
        ax.bar(br2, correlation_f[1], color ='g', width = barWidth,
            edgecolor ='grey', label ='Effort')    
        ax.bar(br3, correlation_f[2], color ='b', width = barWidth,
           edgecolor ='grey', label ='Shape')
        ax.bar(br4, correlation_f[3], color ='c', width = barWidth,
           edgecolor ='grey', label ='Space')
        ax.set_xlabel('window of 20 time_frames ',  fontweight ='bold', fontsize = 17)
        ax.set_ylabel('coefficient_values',  fontweight ='bold', fontsize = 17)
        ax.set_xticks(br1)
        ax.set_xticklabels(v,fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize = 15)
        ax.set_title('Similarity of each LMA component: '+model_name, fontsize = 19) # b/w gen and gnd
        plt.savefig("results/lma_feats/figures/corrplot_all_"+model_name+n_frames+".png")
        plt.clf() # clear plot
        
        fig, ax = plt.subplots(figsize =(16, 9))
        ax.bar(br1, tot_cor, color ='k', width = barWidth,
            edgecolor ='grey')
        ax.set_xlabel('window of 20 time_frames',  fontweight ='bold', fontsize = 17)
        ax.set_ylabel('cor_weighted_sum',  fontweight ='bold', fontsize = 17)
        ax.set_xticks(br1)
        ax.set_xticklabels(v)
        ax.set_title('Weighted sum of correlations: '+model_name, fontsize = 19) # overall correlations
        #fig.suptitle('LMA feature of '+ model_name +' model', fontsize = 25)
        plt.savefig("results/lma_feats/figures/overall_corr_all_"+model_name+n_frames+".png")
        plt.clf() # clear plot
