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


import csv
import pandas as pd


#filename='13m_AllSingle_TrainSplit_13mAllSingle_prot_bert_esm_5e-5_ep1selection15pad_output.csv'
#filename='13m_AllSingle_TrainSplit_13mAllSingle_prot_bert_esm_5e-5_ep2selection15pad_output.csv'
#filename='13m_AllSingle_TrainSplit_13mAllSingle_prot_bert_esm_5e-5_ep3selection_output.csv'
#filename='13m_AllSingle_TrainSplit_13mAllSingle_prot_bert_esm_3e-5_ep1selection15pad_train89_output.csv'
#filename='13m_All89_TrainSplit_13mAllSingle_prot_bert_esm_graph_3e-5_ep1selection_train89_output.csv'
#filename='13m_AllSingle_TrainSplit_13mAllSingle_prot_bert_esm_1e-5_ep3selectionSMALLEST15pad_output.csv'
#filename='13m_All89_TrainSplit_13mAllSingle_prot_bert_esm_graph_3e-5_ep1selection_train89WithPAD_output.csv'

filename='13m_AllSingle_TrainSplit_13mAllSingle_prot_bert_esm_5e-5_ep1selection15pad_output.csv'
#filename='13m_AllSingle_TrainSplit_13mAllSingle_prot_bert_esm_5e-5_ep2selection15pad_output.csv'
#filename='13m_AllSingle_TrainSplit_13mAllSingle_prot_bert_esm_5e-5_ep3selection15pad_output.csv'
#filename='13m_AllSingle_TrainSplit_13m_esm_5e-5_ep2selection15pad_output.csv'


df=pd.read_csv(filename)

if 'Unnamed: 0' in df.columns:
  df=df.drop(['Unnamed: 0'],axis=1)

print(df)

all_original_index=df['original_index'].unique().tolist()

print(len(all_original_index))


data=df.values
#data=[item for item in csv.reader(open(filename, "r",encoding='utf-8'))]
original_to_new_index_mapping_dic={}

for i in range(data.shape[0]):
  #print(i)
  new_index=i
  original_index=data[i][0]
  if original_index not in original_to_new_index_mapping_dic:
    original_to_new_index_mapping_dic[original_index]=[]
  
  original_to_new_index_mapping_dic[original_index].append(new_index)

#print(original_to_new_index_mapping_dic[10741321])




selected_original_index=[]

for nn,ind in enumerate(all_original_index):
  
  print(nn)
  
  all_mhc=df.loc[original_to_new_index_mapping_dic[ind],:]
  #all_mhc=all_mhc.sort_values(by='13mAllSingle_prot_bert_esm_5e-5_ep1selection15pad', ascending=False)
  #all_mhc=all_mhc.sort_values(by='13mAllSingle_prot_bert_esm_5e-5_ep2selection15pad', ascending=False)
  #all_mhc=all_mhc.sort_values(by='13mAllSingle_prot_bert_esm_5e-5_ep3selection', ascending=False)
  #all_mhc=all_mhc.sort_values(by='13mAllSingle_prot_bert_esm_3e-5_ep1selection15pad_train89', ascending=False)
  #all_mhc=all_mhc.sort_values(by='13mAllSingle_prot_bert_esm_graph_3e-5_ep1selection_train89', ascending=False)
  #all_mhc=all_mhc.sort_values(by='13mAllSingle_prot_bert_esm_1e-5_ep3selectionSMALLEST15pad', ascending=False)
  #all_mhc=all_mhc.sort_values(by='13mAllSingle_prot_bert_esm_graph_3e-5_ep1selection_train89WithPAD', ascending=False)
  
  all_mhc=all_mhc.sort_values(by='13mAllSingle_prot_bert_esm_5e-5_ep1selection15pad', ascending=False)
  #all_mhc=all_mhc.sort_values(by='13mAllSingle_prot_bert_esm_5e-5_ep2selection15pad', ascending=False)
  #all_mhc=all_mhc.sort_values(by='13mAllSingle_prot_bert_esm_5e-5_ep3selection15pad', ascending=False)
  #all_mhc=all_mhc.sort_values(by='13m_esm_5e-5_ep2selection15pad', ascending=False)
  
  selected_original_index.append(all_mhc.index[0])
  
  
df_selected=df.loc[selected_original_index,:]

print(df_selected)
  
#df_selected.to_csv('13m_AllSingle_TrainSplit_15padep2start.csv')
#df_selected.to_csv('13m_AllSingle_TrainSplit_15padep3start.csv')
#df_selected.to_csv('13m_AllSingle_TrainSplit_ep4start.csv')
#df_selected.to_csv('13m_AllSingle_TrainSplit_15padtrain89ep2start.csv')
#df_selected.to_csv('13m_All89_TrainSplit_train89graphep2start.csv')
#df_selected.to_csv('13m_AllSingle_TrainSplit_15padep4start_largest.csv') # in ep4, select largest again, to make use of even more samples
#df_selected.to_csv('13m_All89_TrainSplit_train89WithPADgraphep2start.csv')

#df_selected.to_csv('13m_esmOld_5e-5_ep2start.csv')
df_selected.to_csv('13m_esmOld_5e-5_ep3start.csv')
#df_selected.to_csv('13m_esmOld_5e-5_ep4start.csv')
#df_selected.to_csv('13m_esm_5e-5_ep3start.csv')





