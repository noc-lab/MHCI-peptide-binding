

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
from torch.nn import CrossEntropyLoss, MSELoss, LSTM
from transformers.modeling_outputs import SequenceClassifierOutput
import scipy



'''This is a demo to predict the binding probability of an MHC-peptide sequence pair.

To efficiently run the prediction in a batch manner, please refer to "esm_classification.py", which was used by us to predict the whole test set in our paper.'''




_, alphabet = esm.pretrained.esm2_t33_650M_UR50D() # for ESM2-650M model
#_, alphabet = esm.pretrained.esm2_t36_3B_UR50D() # for ESM2-3B model

batch_converter = alphabet.get_batch_converter()




class our_esm(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

        self.classifier = torch.nn.Linear(_.embed_dim, config.num_labels)

        self.esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D() # for ESM2-650M model
        #self.esm, alphabet = esm.pretrained.esm2_t36_3B_UR50D() # for ESM2-3B model

        for param in self.bert.parameters():
            param.requires_grad = False
        
        for param in self.esm.parameters():
            param.requires_grad = True


        
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
        g=None,
        batch_tokens=None,
        seq_len=None
    ):


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        for param in self.esm.parameters():
            param.requires_grad = True


        
        
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

                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        if not return_dict:
            output = (logits,) + outputs2[2:]
            return ((loss,) + output) if loss is not None else output2

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
        )




model=our_esm.from_pretrained('results_13mAllSingle_prot_bert_esm_5e-5_ep2selection15pad/checkpoint-45972/')#.cuda()   # load the model checkpoint, which can be downloaded through the links in this repo
model.eval()


seq='Y Y A M Y Q E N V A Q T D V D T L Y I I Y R D Y T W A A Q A Y R W Y <unk> A A M E P L T A R' # input sequence pair to be predicted, separated by <unk>
seq = re.sub(r"[UZOB]", "X", seq)

tokenizer=AutoTokenizer.from_pretrained('prot_bert_bfd', do_lower_case=False)

seq_ids = tokenizer(seq, truncation=True, padding='max_length', max_length=128)
batch_labels, batch_strs, batch_tokens = batch_converter([("protein1", ''.join(seq.split()))])

seq_len=batch_tokens[0].shape[0]

sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
sample['labels'] = torch.tensor([0])#.cuda()  # just a placeholder for label input

sample['seq_len']=seq_len

sample['batch_tokens'] = torch.nn.functional.pad(batch_tokens[0], (0,52-seq_len), mode='constant', value=1).unsqueeze(0)#.cuda()
sample['g']= sample['labels']+3


logits=model(**sample).logits.cpu().detach().numpy()

predicted_prob=scipy.special.softmax(logits, axis=1)


print('binding probability:',predicted_prob[0,1]) # different devices can generate slightly different results








