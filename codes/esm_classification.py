import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, BertForSequenceClassification
from torch.utils.data import Dataset
import os
import pandas as pd
import requests
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import re
import esm
import csv
import scipy


os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
#os.environ['CUDA_VISIBLE_DEVICES']='2'

#task_name = '13mAllSingle_prot_bert_esm_5e-5_ep1selection' # use mhc training strategy, train only the peptides with 1 mhc in the first epoch
task_name = '13mAllSingle_prot_bert_esm_5e-5_ep1selection15pad'
#task_name = '13mAllSingle_prot_bert_esm_5e-5_ep2selection15pad'
#task_name = '13mAllSingle_prot_bert_esm_5e-5_ep3selection15pad'

model_name = "prot_bert_bfd/"
max_length=128
#sep_token='/'
sep_token='<unk>'
#_, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
#_, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
#_, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
_, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#esm_args=_.args
#print(esm_args)
print(_.num_layers)
print(_.embed_dim)



#test_filename='main_test_set_MHC_input_small.csv'
#test_filename='main_test_set_MHC_input.csv'
#test_filename='16_allele_test_input.csv'

#valid_filename='main_test_set_MHC_input.csv'
#test_filename='complete_test_16_allele_input.csv'

#test_filename='main_test_set_MHC_input.csv'
#model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")


#train_filename="13m_AllSingle_ValSplit.csv"
train_filename="13m_AllSingle_TrainSplit.csv"
#train_filename="13m_esmOld_5e-5_ep2start.csv"
#train_filename="13m_esmOld_5e-5_ep3start.csv"



valid_filename="13m_AllSingle_ValSplit.csv"

#test_filename='hard_test_tet.csv'

#test_filename='easy_test_set.csv'
#test_filename='len_10_test_small.csv'

#test_filename='new_len9_test_set_based_on_13m.csv'


#test_filename='new3_test_set_fullseq_all_allele_after2020_Processed_ForPreviousPos.csv'
#test_filename='test_set_in_netmhc_last_version.csv'
#test_filename='13m_AllSingle_TrainSplit.csv' # output the predicted prob after the 1st epoch for future selection




#test_folder='/data2/mhc/test4_merge_data_processed_esm2small/'
#test_filenames=[test_folder+i for i in os.listdir(test_folder) if i.find('_output')==-1] # or the previous output files will be predicted again

test_filenames=['13m_AllSingle_TrainSplit.csv']
#test_filenames=['test_set_in_netmhc_last_version.csv']




class DeepLocDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_bert_bfd', max_length=1024):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.datasetFolderPath = 'dataset/'
        self.datasetFolderPath = ''
        #self.trainFilePath = os.path.join(self.datasetFolderPath, 'new_train_set.csv')
        #self.trainFilePath = os.path.join(self.datasetFolderPath, 'Russian_data_shuffled_NegUpSampled.csv')
        #self.trainFilePath = os.path.join(self.datasetFolderPath, 'new_train_set_with_more_negative.csv')
        self.trainFilePath = os.path.join(self.datasetFolderPath, train_filename)
        
        self.validFilePath = os.path.join(self.datasetFolderPath, valid_filename)
        #self.testFilePath = os.path.join(self.datasetFolderPath, test_filename)

        # self.downloadDeeplocDataset() # we use our data files

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        
        self.alphabet = alphabet
        self.batch_converter = self.alphabet.get_batch_converter()

        if split=="train":
          #self.seqs, self.labels = self.load_dataset(self.trainFilePath, IsEp1TrainSet=True) # only do this for ep1
          self.seqs, self.labels = self.load_dataset(self.trainFilePath, IsEp1TrainSet=False)
          
          #self.seqs, self.labels = self.load_dataset(self.trainFilePath, IsEp1TrainSet=False)
          print(self.seqs[:15])
          print(self.labels[:15])
        elif split=="valid":
          self.seqs, self.labels = self.load_dataset(self.validFilePath)
        else:
          #self.seqs, self.labels = self.load_dataset(self.testFilePath)
          self.seqs, self.labels = self.load_dataset(split)

        self.max_length = max_length


    def load_dataset(self,path, IsEp1TrainSet=False):
        #df = pd.read_csv(path,names=['ID','MHC_seq','peptide_seq','binding_label','affinity_val','inequality'],skiprows=1)
        #df = pd.read_csv(path,names=['ID','MHC_seq','peptide_seq','binding_label'],skiprows=1)
        df = pd.read_csv(path)
        
        if IsEp1TrainSet==True: # for epoch1 only use these peptides
          df=df[df['num_mhc']==1]
        
        df = df.loc[df['binding_label'].isin([0,1])]
        self.labels_dic = {0:'not_binded',
                           1:'binded'}

        df['labels'] = np.where(df['binding_label']==1, 1, 0)
        
        seq1 = list(df['MHC_seq'])
        seq2 = list(df['peptide_seq'])
        
        seq=[' '.join(seq1[i].upper())+' '+sep_token+' '+' '.join(seq2[i].upper()) for i in range(len(seq1))] # .upper(): there were some strange lower cased AA, hopefully not too many since in old esm1b they would be recognized as <unk>
        if task_name == 'paccman':
            seq = ["".join(s.split()) for s in seq]
        label = list(df['labels'])

        assert len(seq) == len(label)
        return seq, label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #seq = " ".join("".join(self.seqs[idx].split())) # avoid [SEP] becomes [ S E P ], or paccman roberta having extra space
        seq = self.seqs[idx]
        seq = re.sub(r"[UZOB]", "X", seq)
        
        #print(seq)
        
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
        batch_labels, batch_strs, batch_tokens = self.batch_converter([("protein1", ''.join(seq.split()))])
        #print(seq_ids)
        #print(batch_tokens[0])
        #print(batch_tokens[0].shape[0])
        
        seq_len=batch_tokens[0].shape[0]
        
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])
        
        sample['seq_len']=seq_len
        
        #sample['batch_tokens'] = batch_tokens[0]
        #sample['batch_tokens'] = torch.nn.functional.pad(batch_tokens[0], (0,47-seq_len), mode='constant', value=1) # the <pad> token in esm1b has id=1
        sample['batch_tokens'] = torch.nn.functional.pad(batch_tokens[0], (0,52-seq_len), mode='constant', value=1) # the 13m dataset has len8-15, so we pad everything to 34+15+3=52, instead of 47 in len10 before
        
        #print(sample['batch_tokens'])
        sample['g']= torch.tensor(self.labels[idx])+3  # an example of the g we would add
        
        #print(sample)
        
        return sample




train_dataset = DeepLocDataset(split="train", tokenizer_name=model_name, max_length=max_length)
val_dataset = DeepLocDataset(split="valid", tokenizer_name=model_name, max_length=max_length)
#test_dataset = DeepLocDataset(split="test", tokenizer_name=model_name, max_length=max_length)





def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    #prodict_prob = pred.predictions[:,1]
    prodict_prob = scipy.special.softmax(pred.predictions, axis=1)[:,1]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels,prodict_prob)
    # compute PPV like Dima's group
    prodict_prob_list=list(prodict_prob)
    labels_list=list(labels)
    top_ind=sorted(range(len(prodict_prob_list)), key=lambda k: prodict_prob_list[k], reverse=True)
    #top_ind=top_ind[:int(0.1*len(labels_list))] # set the top ratio
    top_ind=top_ind[:int(np.sum(labels_list))] # using the top (# gt positive sample) confident predictions to compute PPV
    top_pred_gt_labels=[labels[i] for i in top_ind]
    print(top_pred_gt_labels)
    ppv=np.sum(top_pred_gt_labels)/len(top_pred_gt_labels)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'PPV': ppv
    }



from torch.nn import CrossEntropyLoss, MSELoss, LSTM
from transformers.modeling_outputs import SequenceClassifierOutput

from esm.pretrained import load_model_and_alphabet_local
#from esm.model import ProteinBertModel

# original class position: '/home/boranhao/anaconda3/envs/torch16/lib/python3.8/site-packages/transformers/models/bert/modeling_bert.py'

class our_esm(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        #self.classifier = torch.nn.Linear(512, config.num_labels) # to set the lstm output dimension as the linear output input.
        self.classifier = torch.nn.Linear(_.embed_dim, config.num_labels)
        #self.classifier = torch.nn.Linear(2560, config.num_labels)
        #print(config.num_labels)
        
        #self.esm, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        #self.esm, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        #self.esm=ProteinBertModel(esm_args,alphabet)
        
        #self.esm, alphabet = load_model_and_alphabet_local("/home/boranhao/.cache/torch/hub/checkpoints/esm1b_t33_650M_UR50S.pt")
        
        #print(self.esm.args)
        
        for param in self.bert.parameters():
            param.requires_grad = False  # the way to freeze the bert layer!!!!!!!!!!
        
        for param in self.esm.parameters():
            param.requires_grad = True  # the way to freeze the bert layer!!!!!!!!!!
        
        #self.esm=self.esm.train()
        
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        g=None, # pass the extra parameters to the network!!!!!!!!!!!!!! g : the graph we need
        batch_tokens=None,
        seq_len=None # pass seq len to truncate seqs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        for param in self.esm.parameters():
            param.requires_grad = True  # the way to freeze the bert layer!!!!!!!!!!
        #self.esm=self.esm.eval()
        
        '''outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )'''
        
        outputs2 = self.esm(
            batch_tokens, 
            [_.num_layers], 
            False
        )
        

        
        hn=outputs2["representations"][_.num_layers][:,0,:] # in 13m data since the seq len difference is larger, we use the [CLS] token instead of average suggested by Facebook


        pooled_output = self.dropout(hn)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                #loss_fct = CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]))
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                '''beta=2
                lgs=logits.view(-1, self.num_labels)
                lgs=torch.nn.functional.softmax(lgs,dim=1)
                #print(lgs)
                #print(lgs.shape)
                prob=lgs[:,1]
                
                #print(prob)
                #print(prob.shape)
                lbs=labels.view(-1).to(torch.float32)
                #print(lbs)
                #print(lbs.shape)
                
                
                F_Beta_index=(1+beta**2)*torch.inner(prob,lbs)/((1+beta**2)*torch.inner(prob,lbs)+(beta**2)*torch.inner(1-prob,lbs)+torch.inner(prob,1-lbs))
                loss=1-F_Beta_index'''

        if not return_dict:
            output = (logits,) + outputs2[2:]
            return ((loss,) + output) if loss is not None else output2

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
        )
    

def model_init():

  
  
  m=our_esm.from_pretrained(model_name)
  #mm, _ = esm.pretrained.esm1b_t33_650M_UR50S()
  #mm, _ = esm.pretrained.esm2_t36_3B_UR50D()
  mm, _ = esm.pretrained.esm2_t33_650M_UR50D()
  m.esm=mm
  return m
  
  

  #return our_esm.from_pretrained('results_13mAllSingle_prot_bert_esm_5e-5_ep1selection15pad/checkpoint-15194/')
  #return our_esm.from_pretrained('results_13mAllSingle_prot_bert_esm_5e-5_ep2selection15pad/checkpoint-45972/')
  #return our_esm.from_pretrained('results_13mAllSingle_prot_bert_esm_5e-5_ep3selection15pad/checkpoint-45972/')
  
  #return our_esm()



training_args = TrainingArguments(
    output_dir='./results_'+task_name,          # output directory
    num_train_epochs=1,              # total number of training epochs # 1st epoch of selection strategy
    per_device_train_batch_size=128,   # batch size per device during training
    per_device_eval_batch_size=128,   # batch size for evaluation
    #warmup_steps=1000,               # number of warmup steps for learning rate scheduler
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=200,               # How often to print logs
    do_train=True,                   # Perform training
    do_eval=True,                    # Perform evaluation
    evaluation_strategy="epoch",     # evalute after eachh epoch
    gradient_accumulation_steps=1,  # total number of steps before back propagation
    #fp16=True,                       # Use mixed precision
    #fp16_opt_level="02",             # mixed precision mode
    run_name="ProBert-BFD-MS",       # experiment name
    seed=4,#3,                           # Seed for experiment reproducibility 3x3 # change seed for 3rd epoch
    save_total_limit=20,
    
    #save_steps=50,
    save_strategy='epoch',#"steps"
    
    #lr_scheduler_type='constant',
    
)






trainer = Trainer(
    model_init=model_init,                # the instantiated ?? Transformers model to be trained
    args=training_args,                   # training arguments, defined above
    train_dataset=train_dataset,          # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics = compute_metrics,    # evaluation metrics
)



trainer.train()





for test_filename in test_filenames:

  test_dataset = DeepLocDataset(split=test_filename, tokenizer_name=model_name, max_length=max_length)
  
  predictions, label_ids, metrics = trainer.predict(test_dataset)
  #prodict_prob = pred.predictions[:,1]
  print(metrics)
  print(predictions)
  
  
  import scipy
  
  predicted_prob=scipy.special.softmax(predictions, axis=1)[:,1]
  
  print(predicted_prob)
  
  test_data=[item for item in csv.reader(open(test_filename, "r",encoding='utf-8'))]
  
  out = open(test_filename[:-4]+'_'+task_name+'_output.csv', 'a', newline='',encoding='utf-8')
  csv_write = csv.writer(out, dialect='excel')
  csv_write.writerow(test_data[0]+[task_name])
  
  test_data=test_data[1:]
  
  for i in range(len(test_data)):
    data_line=test_data[i]
    pred_prob=predicted_prob[i]
    data_line.append(pred_prob)
    csv_write.writerow(data_line)
  
  out.close()
  
  
  
  
  
  
  
  
  
  
  


