import torch
import os

class Config(object):
    def __init__(self):
        DB = "PBMC"
        self.use_cuda = True
        self.threads = 1

        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
            
        elif DB == "PBMC":
            self.number_of_class = 7
            self.input_size = 17668
            self.rna_paths = ['data/citeseq_control_rna.npz']
            self.rna_labels = ['data/citeseq_control_cellTypes.txt']
            self.atac_paths = ['data/asapseq_control_atac.npz']
            self.atac_labels = ['data/asapseq_control_cellTypes.txt']
            self.rna_protein_paths = ['data/citeseq_control_adt.npz']
            self.atac_protein_paths = ['data/asapseq_control_adt.npz']
            self.label_map = 'data/label_to_idx.txt'
            self.label_type_nums=7
            
            # Training config            
            self.batch_size = 128
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 20
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 

            self.alpha = 0.01
            self.lambda_t = 0.6
            self.reg = 0.1
            self.reg_m = 0.1
            self.threshold = 0

        elif DB == "SNARE-seq":
            self.number_of_class = 20
            self.input_size = 16750
            self.rna_paths = ['SNARE-seq/RNA-seq.npz']
            self.rna_labels = ['SNARE-seq/RNA_CellType.txt']
            self.atac_paths = ['SNARE-seq/ATAC-seq.npz']
            self.atac_labels = ['SNARE-seq/ATAC_CellType.txt']
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            self.label_map = 'SNARE-seq/label_to_idx.txt'
            
            # Training config            
            self.batch_size = 128
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 20
            self.epochs_stage3 = 20
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = ''  

            self.alpha = 0.001
            self.lambda_t = 0.6
            self.reg = 0.3
            self.reg_m = 2.5
            self.threshold = 0

