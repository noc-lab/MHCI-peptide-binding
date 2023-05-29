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
import numpy as np
import os
import pandas as pd

def calculate_ppv(df_hits, df_decoys, scores, num_hits, num_decoys, peptide_column, num_iter):
    """Calculates list of ppv scores of length num_iter
        df_hits : pd.Dataframe consisting of hits
        df_decoys : pd.Dataframe consisting of decoys
        scores: dict {peptide:score}
        num_hits: number of hits to sample each iteration
        num_decoys: number of decoys to sample each iteration
        peptide_column: name of peptide column in df_hits 
        num_iter: number of iterations
    """
    
    ppv_scores = []
    for i in range(num_iter):
        iter_df_hits = df_hits.sample(num_hits)
        iter_df_decoys = df_decoys.sample(num_decoys)
        iter_scores = []
        hits_set = set(iter_df_hits[peptide_column])
        decoys_set = set(iter_df_decoys[peptide_column])
        for peptide in hits_set:
            iter_scores.append((scores[peptide], peptide))
        for peptide in decoys_set:
            iter_scores.append((scores[peptide], peptide))
        iter_scores.sort()
        iter_scores.reverse()
        actual_binders = 0
        for i in range(num_hits):
            cur_pep = iter_scores[i][1]
            if cur_pep in hits_set:
                actual_binders += 1
            else:
                assert(cur_pep in decoys_set)
        ppv_score = actual_binders / num_hits
        ppv_scores.append(ppv_score)
    return ppv_scores


def calculate_ppv_df(df, num_iter=100, verbose=False, ratio=100, score_col_name = 'score'):
    """
    Calculates ppv on a DataFrame df.
    To work:
     df should have 'binding','allele','peptide' and 'score' columns;
    'binding' must be integer (0 or 1);
    'allele' must be string;
    'peptide' must be string;
    'score' must be prediction score being of numeric type.
    Inputs:
    df: DataFrame with true labels and predictions
    num_iter: number of iterations
    verbose: print processed alleles
    Returns: dict {allele:ppv_scores}, with ppv_scores being list of ppv samples of size num_iter.
    allele will be present in dict keys iff for allele min(# of positives, (# of negatives)//100)>=1
    """
    df_hits = df[df['binding']==1]
    df_decoys = df[df['binding']==0]
    if type(num_iter)!=int or num_iter<1:
        raise ValueError("num_iter should be positive integer.")
    ppv_scores = dict()
    for allele in df['allele'].unique():
        if verbose:
            print(f'Calculating PPV for {allele}:')
        allele_df_hits = df_hits[df_hits['allele']==allele]
        allele_df_decoys = df_decoys[df_decoys['allele']==allele]
        allele_df = pd.concat([allele_df_hits, allele_df_decoys], ignore_index=True)
        val = min(len(allele_df_hits), len(allele_df_decoys)//ratio)
        if val==0:
            if verbose:
                print(f'Not enough data for {allele}; continue')
            continue
        num_hits = val
        num_decoys = val*ratio
        peptide_to_score = allele_df[['peptide', score_col_name]].set_index('peptide', drop=True)[score_col_name].to_dict()
        ppv_scores[allele] = calculate_ppv(allele_df_hits, allele_df_decoys, peptide_to_score, 
                                            num_hits, num_decoys,'peptide',num_iter)
    return ppv_scores

