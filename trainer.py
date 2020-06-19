from model import UNet
import torch
import torch.nn as nn
import numpy as np
import os
import time
import datetime
import scipy

class Trainer(object):
    """Trainer for training and testing the model"""

    def __init__(self, data_loader, config):
        """Initialize configurations"""

        # model configuration
        self.in_dim = config.in_dim
        self.out_dim = config.out_dim
        self.num_filters = config.num_filters
        self.patch_size = config.patch_size

        # training configuration
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay
        self.resume_iters = config.resume_iters
        self.mode = config.mode

        # miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(config.device_id) \
                                   if self.use_cuda else 'cpu')

        # training result configuration
        self.log_dir = config.log_dir
        self.log_step = config.log_step
        self.model_save_dir = config.model_save_dir
        self.model_save_step = config.model_save_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # data loader
        if self.mode == 'train' or self.mode == 'test':
            self.data_loader = data_loader
        else:
            self.train_data_loader, self.test_data_loader = data_loader

    def build_model(self):
        """Create a model"""
        self.model = UNet(self.in_dim, self.out_dim, self.num_filters)
        self.model = self.model.float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, [self.beta1, self.beta2], weight_decay=self.weight_decay)
        self.print_network(self.model, 'unet')
        self.model.to(self.device)

    def _load(self, checkpoint_path):
        if self.use_cuda:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def restore_model(self, resume_iters):
        """Restore the trained model"""

        print('Loading the trained models from step {}...'.format(resume_iters))
        model_path = os.path.join(self.model_save_dir, '{}-unet'.format(resume_iters)+'.ckpt')
        checkpoint = self._load(model_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def print_network(self, model, name):
        """Print out the network information"""

        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def print_optimizer(self, opt, name):
        """Print out optimizer information"""

        print(opt)
        print(name)

    def build_tensorboard(self):
        """Build tensorboard for visualization"""

        from logger import Logger
        self.logger = Logger(self.log_dir)

    def reset_grad(self):
        """Reset the gradient buffers."""

        self.optimizer.zero_grad()

    def train(self):
        """Train model"""
        if self.mode != 'train_test':
            data_loader = self.data_loader
        else:
            data_loader = self.train_data_loader

        print("current dataset size: ", len(data_loader))
        data_iter = iter(data_loader)

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print('Resuming ...')
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)
            self.print_optimizer(self.optimizer, 'optimizer')

        # print learning rate information
        lr = self.lr
        print('Current learning rates, g_lr: {}.'.format(lr))

        # start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # fetch batch data
            try:
                in_data, label = next(data_iter)
            except:
                data_iter = iter(data_loader)
                in_data, label, _, _, _ = next(data_iter)

            in_data = in_data.float().to(self.device)
            label = label.to(self.device)

            # train the model
            self.model = self.model.train()
            y_out = self.model(in_data)
            loss = nn.BCEWithLogitsLoss() 
            output = loss(y_out, label)
            self.reset_grad()
            output.backward()
            self.optimizer.step()

            # logging
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                log += ", {}: {:.4f}".format("loss", output.mean().item())
                print(log)

                if self.use_tensorboard:
                    self.logger.scalar_summary("loss", output.mean().item(), i+1)

            # save model checkpoints
            if (i+1) % self.model_save_step == 0:
                path = os.path.join(self.model_save_dir, '{}-unet'.format(i+1)+'.ckpt')
                torch.save({'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}, path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def test(self):
        """Test model"""

        if self.mode != 'train_test':
            data_loader = self.data_loader
        else:
            data_loader = self.test_data_loader
        print("current dataset size: ", len(data_loader))
        data_iter = iter(data_loader)

        # start testing on trained model
        if self.resume_iters and self.mode != 'train_test':
            print('Resuming ...')
            self.restore_model(self.resume_iters)

        # start testing.
        result, trace = np.zeros((78,110,24)), np.zeros((78,110,24))
        print('Start testing...')
        correct, total, bcorrect = 0, 0, 0
        while(True):

            # fetch batch data
            try:
                data_in, label, i, j, k = next(data_iter)
            except:
                break

            data_in = data_in.float().to(self.device)
            label = label.float().to(self.device)

            # test the model
            self.model = self.model.eval()
            y_hat = self.model(data_in)
            m = nn.Sigmoid()
            y_hat = m(y_hat)
            y_hat = y_hat.squeeze().detach().cpu().numpy()

            label = label.cpu().numpy().astype(int)
            y_hat_th = (y_hat > 0.2)
            label = (label > 0.5)
            test = (label==y_hat_th)
            correct += np.sum(test)
            btest = (label==0)
            bcorrect += np.sum(btest)
            total += y_hat_th.size
            
            radius = int(self.patch_size / 2)
            for step in range(self.batch_size):
                x, y, z, pred = i[step], j[step], k[step], np.squeeze(y_hat_th[step,:,:,:])
                result[x-radius:x+radius, y-radius:y+radius, z-radius:z+radius] += pred
                trace[x-radius:x+radius, y-radius:y+radius, z-radius:z+radius] += np.ones((self.patch_size,self.patch_size,self.patch_size))

        print('Accuracy: %.3f%%' % (correct/total*100))
        print('Baseline Accuracy: %.3f%%' % (bcorrect/total*100))

        trace += (trace == 0)
        result = result / trace
        scipy.io.savemat('prediction.mat', {'result':result})
        

    def train_test(self):
        """Train and test model"""

        self.train()
        self.test()

