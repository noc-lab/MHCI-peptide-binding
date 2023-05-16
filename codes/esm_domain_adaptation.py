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
import random
import scipy

random.seed(2)

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
print(_.alphabet_size)
print(_.mask_idx)




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
test_filename='test_set_in_netmhc_last_version.csv'
#test_filename='13m_AllSingle_TrainSplit.csv' # output the predicted prob after the 1st epoch for future selection



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
        self.testFilePath = os.path.join(self.datasetFolderPath, test_filename)

        # self.downloadDeeplocDataset() # we use our data files

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        
        self.alphabet = alphabet
        self.batch_converter = self.alphabet.get_batch_converter()

        if split=="train":
          #self.seqs, self.labels = self.load_dataset(self.trainFilePath, IsEp1TrainSet=True) # only do this for ep1
          self.seqs, self.labels = self.load_dataset(self.trainFilePath, IsEp1TrainSet=False) # we never do above in mlm since we want to train over all mhc bags
          
          #self.seqs, self.labels = self.load_dataset(self.trainFilePath, IsEp1TrainSet=False)
          print(self.seqs[:15])
          print(self.labels[:15])
        elif split=="valid":
          self.seqs, self.labels = self.load_dataset(self.validFilePath)
        else:
          self.seqs, self.labels = self.load_dataset(self.testFilePath)

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
        
        sample['batch_tokens_target'] = torch.nn.functional.pad(batch_tokens[0], (0,52-seq_len), mode='constant', value=1) # the 13m dataset has len8-15, so we pad everything to 34+15+3=52, instead of 47 in len10 before
        
        unmasked_seq_pair=torch.nn.functional.pad(batch_tokens[0], (0,52-seq_len), mode='constant', value=1)
        masked_seq_pair=torch.nn.functional.pad(batch_tokens[0], (0,52-seq_len), mode='constant', value=1)
        batch_mlm_masks=torch.LongTensor([0 for i in range(len(masked_seq_pair))])
        #print(unmasked_seq_pair)
        #aa

        non_special_index=[i for i in range(len(unmasked_seq_pair)) if unmasked_seq_pair[i] not in [0,1,2,3]] # 0,1,2,3 are special tokens bos, pad, eos and unk
        masked_index=random.sample(non_special_index,7)
        for i in masked_index:
          if random.randint(1,10)<8.5: # for 80% prob, mask with <mask> (id 32), 20% with original token
            masked_seq_pair[i]=self.alphabet.mask_idx # mask them with special tokens
          batch_mlm_masks[i]=1 # "1" indicates that the position is masked, which will be predicted in esm
        
        
        '''print(sample['batch_tokens_target'])
        print(non_special_index)
        print(masked_index)
        print(batch_mlm_masks)
        print(masked_seq_pair)
        aa'''
        sample['batch_tokens']=masked_seq_pair
        sample['batch_mlm_masks']=batch_mlm_masks
        
        #print(sample['batch_tokens'])
        sample['g']= torch.tensor(self.labels[idx])+3  # an example of the g we would add
        
        #print(sample)
        
        return sample




train_dataset = DeepLocDataset(split="train", tokenizer_name=model_name, max_length=max_length)
val_dataset = DeepLocDataset(split="valid", tokenizer_name=model_name, max_length=max_length)
test_dataset = DeepLocDataset(split="test", tokenizer_name=model_name, max_length=max_length)





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
        seq_len=None, # pass seq len to truncate seqs
        batch_tokens_target=None, # unmaksed sequence pair
        batch_mlm_masks=None, # record the mlm masks


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
        
        
        
        #hn=outputs2["representations"][33][:,1:-1,:].mean(1)
        
        esm_logits=outputs2["logits"]
        #print(seq_len)
        
        '''hn_list=[]
        for b in range(outputs2["representations"][33].shape[0]):
          s_len=seq_len[b]
          hn_list.append(outputs2["representations"][33][b,1:s_len-1,:].mean(0))
        hn=torch.stack(hn_list)'''
        
        hn=outputs2["representations"][_.num_layers][:,0,:] # in 13m data since the seq len difference is larger, we use the [CLS] token instead of average suggested by Facebook
        #print(hn.shape)
        
        #print(outputs)
        #print(outputs[0].shape)
        #print(outputs[1].shape)

        pooled_output = self.dropout(hn)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                #loss_fct = CrossEntropyLoss()
                #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
                loss_fct_mlm = CrossEntropyLoss(reduce=False) # output for each amino acid to account for mlm masks
                loss_mlm = loss_fct_mlm(esm_logits.view(-1, self.esm.alphabet_size), batch_tokens_target.view(-1)) # 33 is the esm alphabet size
                
                #print(loss_mlm.shape)
                #print(loss_mlm)
                
                flattened_mlm_mask=batch_mlm_masks.view(-1)
                #print(flattened_mlm_mask.shape)
                loss_mlm=loss_mlm*flattened_mlm_mask
                #print(loss_mlm)
                
                loss_mlm=torch.sum(loss_mlm)/torch.sum(flattened_mlm_mask) # average only those masked positions
                #print(loss_mlm)
                #aa
                
                loss=loss_mlm
                #loss=loss_mlm+loss

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
  
  
  
  #return our_esm()



training_args = TrainingArguments(
    output_dir='./results_'+task_name,          # output directory
    num_train_epochs=3,              # total number of training epochs # 1st epoch of selection strategy
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
    seed=3,#3,                           # Seed for experiment reproducibility 3x3 # change seed for 3rd epoch
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








