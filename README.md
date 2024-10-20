# MHCI-peptide-binding

## Improved Prediction of MHC-peptide Binding using Protein Language Models
This repository is associated with the paper titled "Improved Prediction of MHC-peptide Binding using Protein Language Models." In the study, we explore the application of deep learning models pretrained on large datasets of protein sequences to predict MHC Class I-peptide binding.

Our models have been evaluated using standard performance metrics in this field, as well as the same training and test sets. The results demonstrate that our models outperform NetMHCpan4.1, which is currently considered the state-of-the-art method.

## Dataset
The training and test sets used in this study can be found at: https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/. For further details regarding the dataset, please refer to the publication "Nucleic Acids Res. 2020 Jul 2;48(W1):W449-W454. doi: 10.1093/nar/gkaa379."

## Pretrained Models
The pretrained models utilized in this research are available at: https://github.com/facebookresearch/esm. 

The checkpoints of our fine-tuned models can be downloaded at:

[ESM1b (domain adaptation + fine-tuning)](https://drive.google.com/drive/folders/1HzM6ZGY8FeCsa1fOY2PDsEZlSLWlu-mO?usp=sharing)

[ESM2-650M (domain adaptation + fine-tuning)](https://drive.google.com/drive/folders/1Z-V-0BVzFi7ScxNDIoxEgq9LaTW6pIBw?usp=sharing)

[ESM2-3B (domain adaptation + fine-tuning)](https://drive.google.com/drive/folders/1kXCfbs6JyCOXkvqmoRfZrdmDRv29c5zj?usp=sharing)

Here is a [demo](https://github.com/noc-lab/MHCI-peptide-binding/blob/main/esm_demo.py) to predict the binding probability of an MHC-peptide sequence pair. To efficiently run the prediction in a batch manner, please refer to "codes/esm_classification.py", which was used by us to predict the whole test set in our paper.
