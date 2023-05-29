# Copyright 2023 Boran Hao, Nasser Hashemi, Dima Kozakov

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#      http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.


import pandas as pd 
import numpy as np 
from sklearn.metrics import precision_recall_curve, roc_auc_score,f1_score, matthews_corrcoef
from sklearn import metrics
from utils import calculate_ppv_df
from tqdm import tqdm 



df = pd.read_csv('./merge_output_36test_all_models_paper_May2023.csv', index_col=0)
grouped = df.groupby(by='allele')


models = ['score_net', 'esm1b_regular',
       'esm2_650_regular', 'esm2_3b_regular', 'esm1b_domain',
       'esm2_650_domain', 'esm2_3b_domain']


################################################################################



# AUC
auc_dict = {}
for n, g in grouped:
    dict_auc_tmp = {}
    for col in models:
        auc_value = roc_auc_score(g['binding'], g[col])
        dict_auc_tmp[col] =  auc_value
    auc_dict[n] = dict_auc_tmp
        

df_auc = pd.DataFrame.from_dict(auc_dict, orient='index')
df_auc = df_auc.round(decimals=3)
print(df_auc.mean().round(decimals=3))

# RP
rp_dict = {}
f1_dict = {}
matt_dict = {}
for n, g in grouped:
    #print(n)
    dict_rp_tmp = {}
    dict_f1_tmp = {}
    dict_matt_tmp = {}
    for col in models:
        #print(col)
        precision, recall, thresholds = precision_recall_curve(g['binding'], g[col])
        auc_pr = metrics.auc(recall, precision)
        
        #n_thresholds = 100
        #thresholds = np.linspace(thresholds.min(), thresholds.max(), n_thresholds)
        
        # Calculate the F1 score for the sampled thresholds
        #f1_scores = []
        #for threshold in thresholds:
        #    y_pred = (g[col] >= threshold).astype(int)
        #    f1_scores.append(f1_score(g['binding'], y_pred))

        #best_threshold = thresholds[np.argmax(f1_scores)]    
        #best_f1_score = max(f1_scores)
        
        best_threshold = .5
        
        
        y_pred_best = (g[col] >= best_threshold).astype(int)
        best_f1_score = f1_score(g['binding'], y_pred_best)
        matthews_value = metrics.matthews_corrcoef(g['binding'], y_pred_best)
        dict_f1_tmp[col] = best_f1_score
        dict_rp_tmp[col] =  auc_pr
        dict_matt_tmp[col] = matthews_value
    rp_dict[n] = dict_rp_tmp
    f1_dict[n] = dict_f1_tmp
    matt_dict[n] =  dict_matt_tmp   

df_pr = pd.DataFrame.from_dict(rp_dict, orient='index')
df_pr = df_pr.round(decimals=3)
print(df_pr.mean().round(decimals=3))

df_f1 = pd.DataFrame.from_dict(f1_dict, orient='index')
df_f1 = df_f1.round(decimals=3)
print(df_f1.mean().round(decimals=3))

df_matt = pd.DataFrame.from_dict(matt_dict, orient='index')
df_matt = df_matt.round(decimals=3)
print(df_matt.mean().round(decimals=3))




#ratio = 99
num_iter = 100

for ratio in range(10,200, 10):
#ppv_samples_us_pos = calculate_ppv_df(df_test_us_pos, verbose=True, ratio=49, score_col_name = 'score')
#for allele in ppv_samples_us_pos.keys():
#    ppv_samples_us_pos[allele] = {"mean":np.mean(ppv_samples_us_pos[allele]), "var":np.var(ppv_samples_us_pos[allele])} 

    ppv_samples_net = calculate_ppv_df(df, verbose=True, num_iter=num_iter, ratio=ratio, score_col_name = 'score_net')
    for allele in ppv_samples_net.keys():
        ppv_samples_net[allele] = {"mean":np.mean(ppv_samples_net[allele]), "std":np.std(ppv_samples_net[allele])} 
     
    
    
    
    ppv_samples_us_esm2_3b = calculate_ppv_df(df, verbose=True, num_iter=num_iter, ratio=ratio, score_col_name = 'esm2_3b_domain')
    for allele in ppv_samples_us_esm2_3b.keys():
        ppv_samples_us_esm2_3b[allele] = {"mean":np.mean(ppv_samples_us_esm2_3b[allele]), "std":np.std(ppv_samples_us_esm2_3b[allele])}
    
    
    compare_results = []
    for k, v in ppv_samples_net.items():
        allele = k
        #our_ppv = v['mean']
        netmhc_ppv = v['mean']
        our_ppv_esm2_3b = ppv_samples_us_esm2_3b[k]['mean']
        
    
        compare_results.append([allele, netmhc_ppv, our_ppv_esm2_3b])
    
    df_compare = pd.DataFrame(data=compare_results, columns = ['Allele', 'NetMHCpan(4.1)', 'ESM2_3B'])
    
    df_compare.to_csv(f'./different_ratio/df_{ratio}.csv')
    

