import torch
import os
import psutil
from datetime import datetime
from config import Config
from training import model_training
from evaluations import evaluate_atac_predictions


def main():    

    process = psutil.Process(os.getpid())
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # initialization 
    config = Config()    
    torch.manual_seed(config.seed)
    print('Start time: ', datetime.now().strftime('%H:%M:%S'))
    
    # training
    print('Training start')
    model = model_training(config)    
    for epoch in range(config.epochs_stage1):
        print('Epoch:', epoch)
        model.train(epoch)
    
    print('Write embeddings')
    model.write_embeddings()
    print('Training finished: ', datetime.now().strftime('%H:%M:%S'))
    

    evaluate_atac_predictions(config)
    print('finished: ', datetime.now().strftime('%H:%M:%S'))
    
    
if __name__ == "__main__":
    main()

