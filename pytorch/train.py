import sys
sys.path.append("..")
import numpy as np
import os
import torch
import torch.nn as nn
import math
from rnnlm import *
from ptb import *
from logger import *
from torch.autograd import Variable

gpu_number = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
torch.manual_seed(24)

class Trainer():
    def __init__(self, train_batch_size=20, train_seq_len=35, test_batch_size=10, test_seq_len=35, exp_name='exp'):
        self.dataset = PTB(train_batch_size, test_batch_size, train_seq_len, test_seq_len)
        self.model = RNNLM(self.dataset.vocabulary_size, 200, 200, 2).cuda()
        self.logger = Logger(exp_dir=exp_name)
        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), 20.0)

    def train(self, epochs=1):
        best_test_loss = None
        for i in range(epochs):
            self.model.train()
            total_loss = 0.0
            hidden = self.to_cuda(self.model.init_hidden(self.dataset.train_batch_size))
            for j in range(self.dataset.train_batch_count):
                batch_inputs, batch_targets = self.dataset.next_train_batch(j)
                batch_inputs = torch.from_numpy(batch_inputs).cuda()
                batch_targets = torch.from_numpy(batch_targets).cuda()
                hidden = self.repackage_hidden(hidden)
                logits, hidden = self.model(batch_inputs, hidden)
                loss = self.criterion(logits.view(-1, self.dataset.vocabulary_size), batch_targets.view(-1))
                total_loss += loss.item()
                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.optimizer.step()
                
            avg_loss = total_loss / self.dataset.train_batch_count 
            # logging training results
            self.logger.log('Training epoch {0}'.format(i))
            self.logger.log('    train loss {0}, train perplexity {1}'.format(avg_loss, np.exp(avg_loss)))
            total_loss = 0.0
            # evaluate on test set
            self.model.eval()
            hidden = self.to_cuda(self.model.init_hidden(self.dataset.test_batch_size))
            for j in range(self.dataset.test_batch_count):
                batch_inputs, batch_targets = self.dataset.next_test_batch(j)
                batch_inputs = torch.from_numpy(batch_inputs).cuda()
                batch_targets = torch.from_numpy(batch_targets).cuda()
                logits, hidden = self.model(batch_inputs, hidden)
                loss = self.criterion(logits.view(-1, self.dataset.vocabulary_size), batch_targets.view(-1))
                total_loss += loss.item()
            avg_loss = total_loss / self.dataset.test_batch_count

            if best_test_loss is not None and best_test_loss < avg_loss:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                for g in self.optimizer.param_groups:
                    g['lr'] /= 4.0
            elif best_test_loss is None or best_test_loss > avg_loss:
                best_test_loss = avg_loss

            # logging validation results
            self.logger.log('    test loss {0}, test perplexity {1}'.format(avg_loss, np.exp(avg_loss)))
            # save model
            save_model_file = os.path.join(self.logger.exp_dir, 'ResNet-model_epoch' + str(i))
            if i % 20 == 0:
                state = {'epoch': i + 1,
                         'state_dict': self.model.state_dict(),
                         'optimizer' : self.optimizer.state_dict()
                        } 
                torch.save(state, save_model_file)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)
    
    def to_cuda(self, h):
        if isinstance(h, torch.Tensor):
            return h.cuda()
        else:
            return tuple(self.to_cuda(v) for v in h)

if __name__ == "__main__":
    trainer = Trainer(exp_name='exp3')
    trainer.train(epochs=100)
