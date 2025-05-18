import torch
import torch.optim as optim
import ot
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import cycle
from scipy.linalg import norm
from scipy.special import softmax
from dataloader import PrepareDataloader
from model import Net_encoder, Net_class
from loss import L1regularization, EncodingLoss, ClassLoss
from utils import *


def prepare_input(data_list, config):
    output = []
    for data in data_list:
        output.append(Variable(data.to(config.device)))
    return output


def def_cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class model_training():
    def __init__(self, config):
        self.config = config
        # load data
        self.train_rna_loaders, self.test_rna_loaders, self.train_atac_loaders, self.test_atac_loaders, self.training_iters = PrepareDataloader(config).getloader()
        self.training_iteration = 0
        for atac_loader in self.train_atac_loaders:
            self.training_iteration += len(atac_loader)
        
        # initialize dataset       
        if self.config.use_cuda:
            self.model_encoder = torch.nn.DataParallel(Net_encoder(config.input_size).to(self.config.device))
            self.model_class = torch.nn.DataParallel(Net_class(config.number_of_class).to(self.config.device))
        else:
            self.model_encoder = Net_encoder(config.input_size).to(self.config.device)
            self.model_class = Net_class(config.number_of_class).to(self.config.device)
                
        # initialize criterion (loss)
        self.criterion_class = ClassLoss()
        self.criterion_encoding = EncodingLoss(dim=64, p=config.p, use_gpu = self.config.use_cuda)
        self.l1_regular = L1regularization()
        
        # initialize optimizer (sgd/momemtum/weight decay)
        self.optimizer_encoder = optim.SGD(self.model_encoder.parameters(), lr=self.config.lr_stage1, momentum=self.config.momentum,
                                           weight_decay=0)
        self.optimizer_class = optim.SGD(self.model_class.parameters(), lr=self.config.lr_stage1, momentum=self.config.momentum,
                                        weight_decay=0)


    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.config.lr_stage1 * (0.1 ** ((epoch - 0) // self.config.lr_decay_epoch))
        if (epoch - 0) % self.config.lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def load_checkpoint(self, args):
        if self.config.checkpoint is not None:
            if os.path.isfile(self.config.checkpoint):
                print("=> loading checkpoint '{}'".format(self.config.checkpoint))
                checkpoint = torch.load(self.config.checkpoint)                
                self.model_encoder.load_state_dict(checkpoint['model_encoding_state_dict'])
                self.model_class.load_state_dict(checkpoint['model_class_state_dict'])
            else:
                print("=> no resume checkpoint found at '{}'".format(self.config.checkpoint))


    def train(self, epoch):
        self.model_encoder.train()
        self.model_class.train()
        total_encoding_loss, total_class_loss, total_transfer_loss = 0., 0., 0.
        self.adjust_learning_rate(self.optimizer_encoder, epoch)
        self.adjust_learning_rate(self.optimizer_class, epoch)

        # initialize iterator
        iter_rna_loaders = []
        iter_atac_loaders = []
        for rna_loader in self.train_rna_loaders:
            iter_rna_loaders.append(def_cycle(rna_loader))
        for atac_loader in self.train_atac_loaders:
            iter_atac_loaders.append(def_cycle(atac_loader))
                
        for batch_idx in range(self.training_iters):
            # rna forward
            rna_embeddings = []
            rna_class_predictions = []
            rna_labels = []
            for iter_rna_loader in iter_rna_loaders:
                rna_data, rna_label = next(iter_rna_loader)    
                # prepare data
                rna_data, rna_label = prepare_input([rna_data, rna_label], self.config)
                # model forward
                rna_embedding = self.model_encoder(rna_data)
                rna_class_prediction = self.model_class(rna_embedding)
                rna_embeddings.append(rna_embedding)
                rna_class_predictions.append(rna_class_prediction)
                rna_labels.append(rna_label)
                
            # atac forward
            atac_embeddings = []
            atac_class_predictions = []
            for iter_atac_loader in iter_atac_loaders:
                atac_data = next(iter_atac_loader)    
                # prepare data
                atac_data = prepare_input([atac_data], self.config)[0]
                # model forward
                atac_embedding = self.model_encoder(atac_data)
                atac_class_prediction = self.model_class(atac_embedding)

                atac_embeddings.append(atac_embedding)
                atac_class_predictions.append(atac_class_prediction)
            
            
            # caculate loss  
            class_loss = self.criterion_class(rna_class_predictions[0], rna_labels[0])
            for i in range(1, len(rna_class_predictions)):
                class_loss += self.criterion_class(rna_class_predictions[i], rna_labels[i])
            class_loss = class_loss/len(rna_class_predictions)
            
            encoding_loss = self.criterion_encoding(atac_embeddings, rna_embeddings)
            regularization_loss_encoder = self.l1_regular(self.model_encoder)            
            
            # OT loss
            M_embed = torch.cdist(rna_embeddings[0], atac_embeddings[0]) ** 2
            rna_labels = F.one_hot(rna_labels[0].long(), num_classes = self.config.number_of_class).float()
            atac_class_predictions = F.softmax(atac_class_predictions[0], dim=1)
            M_sce = -torch.mm(rna_labels, torch.transpose(torch.log(atac_class_predictions), 0, 1))
            M = self.config.alpha * M_embed + self.config.lambda_t * M_sce

            a,b = ot.unif(len(rna_embeddings[0])), ot.unif(len(atac_embeddings[0]))
            pi = ot.unbalanced.mm_unbalanced(a,b,M.detach().cpu().numpy(), reg=self.config.reg, reg_m=self.config.reg_m) 
            pi = torch.from_numpy(pi).float().to(self.config.device)
            transfer_loss = torch.sum(pi * M)

            
            # update encoding weights
            self.optimizer_encoder.zero_grad()  
            regularization_loss_encoder.backward(retain_graph=True)
            encoding_loss.backward(retain_graph=True)
            transfer_loss.backward(retain_graph=True)        
            self.optimizer_encoder.step()
            regularization_loss_class = self.l1_regular(self.model_class)

            # update class weights
            self.optimizer_class.zero_grad()
            class_loss.backward(retain_graph=True)
            regularization_loss_class.backward(retain_graph=True)
            self.optimizer_class.step()

            # print log
            total_encoding_loss += encoding_loss.data.item()
            total_class_loss += class_loss.data.item()
            total_transfer_loss += transfer_loss.item()

            progress_bar(batch_idx, self.training_iters,
                         'encoding_loss: %.3f, rna_loss: %.3f , transfer_loss: %.3f' % (
                             total_encoding_loss / (batch_idx + 1), total_class_loss / (batch_idx + 1),
                             total_transfer_loss / (batch_idx + 1)
                             ))
        # save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'model_class_state_dict': self.model_class.state_dict(),
            'model_encoding_state_dict': self.model_encoder.state_dict(),
            'optimizer': self.optimizer_class.state_dict()            
        })
        
        
    def write_embeddings(self):
        self.model_encoder.eval()
        self.model_class.eval()
        if not os.path.exists("output/"):
            os.makedirs("output/")
        
        # rna db
        for i, rna_loader in enumerate(self.test_rna_loaders):
            db_name = os.path.basename(self.config.rna_paths[i]).split('.')[0]
            fp_em = open('./output/' + db_name + '_embeddings.txt', 'w')
            fp_pre = open('./output/' + db_name + '_predictions.txt', 'w')
            for batch_idx, (rna_data, rna_label) in enumerate(rna_loader):    
                # prepare data
                rna_data, rna_label = prepare_input([rna_data, rna_label], self.config)
                    
                # model forward
                rna_embedding = self.model_encoder(rna_data)
                rna_class_prediction = self.model_class(rna_embedding)
                            
                rna_embedding = rna_embedding.data.cpu().numpy()
                rna_class_prediction = rna_class_prediction.data.cpu().numpy()
                
                # normalization & softmax
                rna_embedding = rna_embedding / norm(rna_embedding, axis=1, keepdims=True)
                rna_class_prediction = softmax(rna_class_prediction, axis=1)
                                
                # write embeddings
                test_num, embedding_size = rna_embedding.shape
                for print_i in range(test_num):
                    fp_em.write(str(rna_embedding[print_i][0]))
                    for print_j in range(1, embedding_size):
                        fp_em.write(' ' + str(rna_embedding[print_i][print_j]))
                    fp_em.write('\n')
                    
                # write predictions
                test_num, prediction_size = rna_class_prediction.shape
                for print_i in range(test_num):
                    fp_pre.write(str(rna_class_prediction[print_i][0]))
                    for print_j in range(1, prediction_size):
                        fp_pre.write(' ' + str(rna_class_prediction[print_i][print_j]))
                    fp_pre.write('\n')
                
                progress_bar(batch_idx, len(rna_loader),
                         'write embeddings and predictions for db:' + db_name)                    
            fp_em.close()
            fp_pre.close()
        
        
        # atac db
        for i, atac_loader in enumerate(self.test_atac_loaders):
            db_name = os.path.basename(self.config.atac_paths[i]).split('.')[0]
            fp_em = open('./output/' + db_name + '_embeddings.txt', 'w')
            fp_pre = open('./output/' + db_name + '_predictions.txt', 'w')
            fp_label = open('./output/' + db_name + '_ot_predictions.txt', 'w')
            for batch_idx, (atac_data) in enumerate(atac_loader):    
                # prepare data
                atac_data = prepare_input([atac_data], self.config)[0]
                
                # model forward
                atac_embedding = self.model_encoder(atac_data)
                atac_class_prediction = self.model_class(atac_embedding)
                                                                
                atac_embedding = atac_embedding.data.cpu().numpy()
                atac_class_prediction = atac_class_prediction.data.cpu().numpy()
                
                # normalization & softmax
                atac_embedding = atac_embedding / norm(atac_embedding, axis=1, keepdims=True)
                atac_class_prediction = softmax(atac_class_prediction, axis=1)
                
                # write embeddings
                test_num, embedding_size = atac_embedding.shape
                for print_i in range(test_num):
                    fp_em.write(str(atac_embedding[print_i][0]))
                    for print_j in range(1, embedding_size):
                        fp_em.write(' ' + str(atac_embedding[print_i][print_j]))
                    fp_em.write('\n')
                    
                # write predictions
                test_num, prediction_size = atac_class_prediction.shape
                for print_i in range(test_num):
                    fp_pre.write(str(atac_class_prediction[print_i][0]))
                    for print_j in range(1, prediction_size):
                        fp_pre.write(' ' + str(atac_class_prediction[print_i][print_j]))
                    fp_pre.write('\n')

                test_num = atac_class_prediction.shape[0]

                atac_class_prediction = torch.tensor(atac_class_prediction)
                max_pred, atac_ot_prediction = torch.max(atac_class_prediction, dim =1)
                atac_ot_prediction[max_pred < self.config.threshold] = -1

                for label in atac_ot_prediction:
                    fp_label.write(f"{label.item()}\n")
                                    
                progress_bar(batch_idx, len(atac_loader),
                         'write embeddings and predictions for db:' + db_name)                    
            fp_em.close()
            fp_pre.close()
            fp_label.close()

