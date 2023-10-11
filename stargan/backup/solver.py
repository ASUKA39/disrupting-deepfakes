from model import Generator, AvgBlurGenerator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime

import attacks_fft_numpy as attacks
# import attacks_fft_numpy_fft as attacks

# import attacks
from pytorch_msssim import ssim


from PIL import ImageFilter
from PIL import Image
from torchvision import transforms

import defenses.smoothing as smoothing

# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
np.random.seed(0)

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        self.build_adv_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            # self.G = AvgBlurGenerator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)

    def build_adv_model(self):
        """Create a adversarial generator and a adversarial discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.adv_G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.adv_D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        elif self.dataset in ['Both']:
            self.adv_G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.adv_D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer2 = torch.optim.Adam(self.adv_G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer2 = torch.optim.Adam(self.adv_D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.adv_G, 'adv_G')
        self.print_network(self.adv_D, 'adv_D')

        self.adv_G.to(self.device)
        self.adv_D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):#设置路径并load G和D的参数
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))

        # self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.load_model_weights(self.G, G_path)
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def restore_adv_model(self, resume_iters):
        """Restore the trained adversarial generator and discriminator."""
        print('Loading the trained adversarial models from step {}...'.format(resume_iters))
        adv_G_path = os.path.join(self.model_save_dir, '{}-adv-G.ckpt'.format(resume_iters))
        adv_D_path = os.path.join(self.model_save_dir, '{}-adv-D.ckpt'.format(resume_iters))

        self.load_model_weights(self.adv_G, adv_G_path)
        self.adv_D.load_state_dict(torch.load(adv_D_path, map_location=lambda storage, loc: storage))

####
    def build_and_restore_alt_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G2 = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D2 = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        elif self.dataset in ['Both']:
            self.G2 = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D2 = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer2 = torch.optim.Adam(self.G2.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer2 = torch.optim.Adam(self.D2.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G2, 'G')
        self.print_network(self.D2, 'D')

        self.G2.to(self.device)
        self.D2.to(self.device)
        """Restore the trained generator and discriminator."""
        resume_iters = 50000
        model_save_dir = 'stargan/models'
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(model_save_dir, '{}-D.ckpt'.format(resume_iters))

        # self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.load_model_weights(self.G2, G_path)
        self.D2.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

####
    def load_model_weights(self, model, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'preprocessing' not in k}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(pretrained_dict, strict=False)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self):
        """Vanilla Training of StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)       # No Attack
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake, _ = self.G(x_real, c_trg)       # No Attack
            out_src, out_cls = self.D(x_fake.detach())  # No Attack
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)  # No Attack
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                x_fake, _ = self.G(x_real, c_trg)     # No Attack
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_reconst, _ = self.G(x_fake, c_org)      # No Attack
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        elt, _ = self.G(x_fixed, c_fixed)
                        x_fake_list.append(elt)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

####
    def train_adv_gen(self):
        """Adversarial Training for StarGAN only for Generator, within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)       # No Attack
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake, _ = self.G(x_real, c_trg)       # No Attack
            out_src, out_cls = self.D(x_fake.detach())  # No Attack
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)  # No Attack
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            # Black image
            black = np.zeros((x_real.shape[0],3,256,256))
            black = torch.FloatTensor(black).to(self.device)

            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)
                x_real_adv = attacks.perturb_batch(x_real, black, c_trg, self.G, pgd_attack)

                x_fake, _ = self.G(x_real_adv, c_trg)   # Attack
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_fake_adv = attacks.perturb_batch(x_fake, black, c_org, self.G, pgd_attack)
                x_reconst, _ = self.G(x_fake_adv, c_org)    # Attack
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        elt, _ = self.G(x_fixed, c_fixed)
                        x_fake_list.append(elt)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

####
    def train_adv_both(self):
        """G+D Adversarial Training for StarGAN with both Discriminator and Generator, within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # Black image
            black = np.zeros((x_real.shape[0],3,256,256))
            black = torch.FloatTensor(black).to(self.device)

            pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)
            x_real_adv = attacks.perturb_batch(x_real, black, c_trg, self.G, pgd_attack)    # Adversarial training

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real_adv)   # Attack
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake, _ = self.G(x_real_adv, c_trg)   # Attack
            x_fake_adv = attacks.perturb_batch(x_fake, black, c_org, self.G, pgd_attack)    # Adversarial training
            out_src, out_cls = self.D(x_fake_adv.detach())  # Attack
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real_adv.data + (1 - alpha) * x_fake_adv.data).requires_grad_(True)  # Attack
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                # Original-to-target domain
                x_fake, _ = self.G(x_real_adv, c_trg)   # Attack
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_fake_adv = attacks.perturb_batch(x_fake, black, c_org, self.G, pgd_attack)
                x_reconst, _ = self.G(x_fake_adv, c_org)    # Attack
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        elt, _ = self.G(x_fixed, c_fixed)
                        x_fake_list.append(elt)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_multi(self):
        """Vanilla Training for StarGAN with multiple datasets."""
        # Data iterators.
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)

        # Fetch fixed inputs for debugging.
        x_fixed, c_org = next(celeba_iter)
        x_fixed = x_fixed.to(self.device)
        c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
        c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
        zero_celeba = torch.zeros(x_fixed.size(0), self.c_dim).to(self.device)           # Zero vector for CelebA.
        zero_rafd = torch.zeros(x_fixed.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
        mask_celeba = self.label2onehot(torch.zeros(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
        mask_rafd = self.label2onehot(torch.ones(x_fixed.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for dataset in ['CelebA', 'RaFD']:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter

                try:
                    x_real, label_org = next(data_iter)
                except:
                    if dataset == 'CelebA':
                        celeba_iter = iter(self.celeba_loader)
                        x_real, label_org = next(celeba_iter)
                    elif dataset == 'RaFD':
                        rafd_iter = iter(self.rafd_loader)
                        x_real, label_org = next(rafd_iter)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'CelebA':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                    zero = torch.zeros(x_real.size(0), self.c2_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                    c_org = torch.cat([c_org, zero, mask], dim=1)
                    c_trg = torch.cat([c_trg, zero, mask], dim=1)
                elif dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)
                    zero = torch.zeros(x_real.size(0), self.c_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                    c_org = torch.cat([zero, c_org, mask], dim=1)
                    c_trg = torch.cat([zero, c_trg, mask], dim=1)

                x_real = x_real.to(self.device)             # Input images.
                c_org = c_org.to(self.device)               # Original domain labels.
                c_trg = c_trg.to(self.device)               # Target domain labels.
                label_org = label_org.to(self.device)       # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)       # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org, dataset)

                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i+1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg, dataset)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training info.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i+1, self.num_iters, dataset)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_celeba_list:
                        c_trg = torch.cat([c_fixed, zero_rafd, mask_celeba], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    for c_fixed in c_rafd_list:
                        c_trg = torch.cat([zero_celeba, c_fixed, mask_rafd], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset. No attack."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg)[0])

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

####
    def test_attack(self):
        """Vanilla or blur attacks."""

        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Initialize Metrics
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

        for i, (x_real, c_org) in enumerate(data_loader):#外层循环，遍历照片
            # Prepare input images and target domain labels.
            x_real = x_real.to(self.device)
            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

            pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)

            # Translated images.
            x_fake_list = [x_real]

            for idx, c_trg in enumerate(c_trg_list):#内层循环，遍历属性
                print('image', i, 'class', idx)
                with torch.no_grad():
                    x_real_mod = x_real
                    # x_real_mod = self.blur_tensor(x_real_mod) # use blur
                    gen_noattack, gen_noattack_feats = self.G(x_real_mod, c_trg)

                # Attacks
                x_adv, perturb = pgd_attack.perturb(x_real, gen_noattack, c_trg)                          # Vanilla attack
                # x_adv, perturb, blurred_image = pgd_attack.perturb_blur(x_real, gen_noattack, c_trg)    # White-box attack on blur
                # x_adv, perturb = pgd_attack.perturb_blur_iter_full(x_real, gen_noattack, c_trg)         # Spread-spectrum attack on blur
                # x_adv, perturb = pgd_attack.perturb_blur_eot(x_real, gen_noattack, c_trg)               # EoT blur adaptation

                # Generate adversarial example
                x_adv = x_real + perturb

                # No attack
                # x_adv = x_real

                # x_adv = self.blur_tensor(x_adv)   # use blur

                # Metrics
                with torch.no_grad():
                    gen, _ = self.adv_G(x_adv.float(), c_trg)

                    # Add to lists
                    # x_fake_list.append(blurred_image)
                    x_fake_list.append(x_adv)
                    # x_fake_list.append(perturb)
                    x_fake_list.append(gen)

                    l1_error += F.l1_loss(gen, gen_noattack)
                    l2_error += F.mse_loss(gen, gen_noattack)
                    l0_error += (gen - gen_noattack).norm(0)
                    min_dist += (gen - gen_noattack).norm(float('-inf'))
                    if F.mse_loss(gen, gen_noattack) > 0.05:
                        n_dist += 1
                    n_samples += 1

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            if i == 49:     # stop after this many images
                break

        # Print metrics
        print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples,
        l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

####
    '''def test_attack_feats(self):
        """Feature-level attacks"""

        # Mapping of feature layers to indices
        layer_dict = {0: 2, 1: 5, 2: 8, 3: 9, 4: 10, 5: 11, 6: 12, 7: 13, 8: 14, 9: 17, 10: 20, 11: None}

        for layer_num_orig in range(12):    # 11 layers + output
            # Load the trained generator.
            self.restore_model(self.test_iters)

            # Set data loader.
            if self.dataset == 'CelebA':
                data_loader = self.celeba_loader
            elif self.dataset == 'RaFD':
                data_loader = self.rafd_loader

            # Initialize Metrics
            l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
            n_dist, n_samples = 0, 0

            print('Layer', layer_num_orig)

            for i, (x_real, c_org) in enumerate(data_loader):
                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                layer_num = layer_dict[layer_num_orig]  # get layer number
                pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=layer_num)

                # Translate images.
                x_fake_list = [x_real]

                for c_trg in c_trg_list:
                    with torch.no_grad():
                        gen_noattack, gen_noattack_feats = self.G(x_real, c_trg)

                    # Attack
                    if layer_num == None:
                        x_adv, perturb = pgd_attack.perturb(x_real, gen_noattack, c_trg)
                    else:
                        x_adv, perturb = pgd_attack.perturb(x_real, gen_noattack_feats[layer_num], c_trg)

                    x_adv = x_real + perturb

                    # Metrics
                    with torch.no_grad():
                        gen, gen_feats = self.G(x_adv, c_trg)

                        # Add to lists
                        x_fake_list.append(x_adv)
                        x_fake_list.append(gen)

                        l1_error += F.l1_loss(gen, gen_noattack)
                        l2_error += F.mse_loss(gen, gen_noattack)
                        l0_error += (gen - gen_noattack).norm(0)
                        min_dist += (gen - gen_noattack).norm(float('-inf'))
                        if F.mse_loss(gen, gen_noattack) > 0.05:
                            n_dist += 1
                        n_samples += 1

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-{}-images.jpg'.format(layer_num_orig, i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                if i == 49:
                    break

            # Print metrics
            print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples,
            l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))'''

####
    def test_attack_cond(self):
        """Class conditional transfer"""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Initialize Metrics
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

        for i, (x_real, c_org) in enumerate(data_loader):
            # Prepare input images and target domain labels.
            x_real = x_real.to(self.device)
            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

            pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)

            # Translate images.
            x_fake_list = [x_real]

            for idx, c_trg in enumerate(c_trg_list):
                print(i, idx)
                with torch.no_grad():
                    x_real_mod = x_real
                    gen_noattack, gen_noattack_feats = self.G(x_real_mod, c_trg)

                # Transfer to different classes
                if idx == 0:
                    # Wrong Class
                    x_adv, perturb = pgd_attack.perturb(x_real, gen_noattack, c_trg_list[0])

                    # Joint Class Conditional
                    # x_adv, perturb = pgd_attack.perturb_joint_class(x_real, gen_noattack, c_trg_list)

                    # Iterative Class Conditional
                    # x_adv, perturb = pgd_attack.perturb_iter_class(x_real, gen_noattack, c_trg_list)

                # Correct Class
                # x_adv, perturb = pgd_attack.perturb(x_real, gen_noattack, c_trg)

                x_adv = x_real + perturb

                # Metrics
                with torch.no_grad():
                    gen, _ = self.G(x_adv, c_trg)

                    # Add to lists
                    x_fake_list.append(x_adv)
                    # x_fake_list.append(perturb)
                    x_fake_list.append(gen)

                    l1_error += F.l1_loss(gen, gen_noattack)
                    l2_error += F.mse_loss(gen, gen_noattack)
                    l0_error += (gen - gen_noattack).norm(0)
                    min_dist += (gen - gen_noattack).norm(float('-inf'))
                    if F.mse_loss(gen, gen_noattack) > 0.05:
                        n_dist += 1
                    n_samples += 1

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            if i == 49:
                break

        # Print metrics
        print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples,
        l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.celeba_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
                c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
                zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(self.device)            # Zero vector for CelebA.
                zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

####用来进行高斯模糊/平均模糊
    def blur_tensor(self, tensor):
        # preproc = smoothing.AverageSmoothing2D(channels=3, kernel_size=9).to(self.device)
        preproc = smoothing.GaussianSmoothing2D(sigma=1.5, channels=3, kernel_size=11).to(self.device)
        return preproc(tensor)

        # 三通道傅立叶变换

    def fft(self, channels):
        '''
        channles: (1,3,256,256)的tensor数组
        return: (1,3,256,256)的tensor数组
        '''

        channels_ = [channels[0][0], channels[0][1], channels[0][2]]
        fchannels = []
        for i in range(3):
            channel = torch.fft.fft2(channels_[i])
            fchannels.append(torch.fft.fftshift(channel))

        f = torch.stack(fchannels,0)
        f = f.reshape(1,3,256,256)
        return f

    def get_amp_ang(self, X):
        '''
        X: (1,3,256,256)tensor数组
        return: 2*(1,3,256,256)的tensor.cuda数组
        '''

        _fft = self.fft(X)
        amp = []
        ang = []
        for i in range(3):
            amp.append(torch.abs(_fft[0][i]))
            ang.append(torch.angle(_fft[0][i]))

        _ang = torch.stack(ang,0)
        _amp = torch.stack(amp,0)

        _ang = torch.unsqueeze(_ang,0)
        _amp = torch.unsqueeze(_amp,0)
        return _amp,_ang

        # 加噪复原

    def recover(self, amp, ang):
        '''
        amp,ang:两个(1,3,256,256)tensor.cuda数组
        return: (1,3,256,256)tensor
        '''
        _re = []
        for i in range(3):
            re = torch.complex(amp[0][i] * torch.cos(ang[0][i]), (amp[0][i])* torch.sin(ang[0][i]))
            re = torch.fft.ifftshift(re)
            re = torch.fft.ifft2(re)
            re = torch.real(re)

            _re.append(re)
        #torch.tensor(X)).to(torch.float32).cuda(), (torch.tensor(per * 100)).to(torch.float32).cuda()

        re = torch.stack(_re,0)
        re = torch.unsqueeze(re,0)

        return re

    def protect(self,now,backup,_range):
        rows, cols = now[0][0].shape

        row, col = rows // 2, cols // 2

        for i in range(3):
            now[0][i][row - _range:row + _range - 1, col - _range:col + _range -1] = backup[0][i][row - _range:row + _range - 1,
                                                                            col - _range:col + _range - 1]
        return now

    def recover_fft(self,amp,ang,backup,center):
        amp_backup,ang_backup = self.get_amp_ang(backup)
        amp = self.protect(amp,amp_backup,center // 2)
        ang = self.protect(ang,ang_backup,center // 2)

        _re = []
        for i in range(3):
            re = torch.complex(amp[0][i] * torch.cos(ang[0][i]), (amp[0][i])* torch.sin(ang[0][i]))
            re = torch.fft.ifftshift(re)
            re = torch.fft.ifft2(re)
            re = torch.real(re)

            _re.append(re)
        #torch.tensor(X)).to(torch.float32).cuda(), (torch.tensor(per * 100)).to(torch.float32).cuda()

        re = torch.stack(_re,0)
        re = torch.unsqueeze(re,0)

        return re

    def compute_luminance_weight(self, X):
        """
        X: (1,3,256,256)Tensor数组
        Return: (1,3,256,256)Tensor数组
        """

        #arg
        sigmoid_scale = 35

        # luminance = 0.299R + 0.587G + 0.114B
        # 用RGB到灰度转换公式计算luminance
        luminance = 0.299 * X[:, 0, :, :] + 0.587 * X[:, 1, :, :] + 0.114 * X[:, 2, :, :]

        # Normalize
        luminance = (luminance - luminance.min()) / (luminance.max() - luminance.min())

        # Use sigmoid to map luminance to a weight.
        weight = torch.sigmoid(sigmoid_scale * (luminance - 0.5))

        return weight

    def perturb_with_luminance(self, X, adversarial_noise):
        """
        X: (1,3,256,256)Tensor数组
        adversarial_noise: 原始对抗样本
        
        Returns: 调整后的对抗样本
        """
        luminance_weight = self.compute_luminance_weight(X)
        luminance_weight = luminance_weight.unsqueeze(1)  # Add channel dimension

        adjusted_noise = adversarial_noise * luminance_weight

        return adjusted_noise

    def psnr(self, img1, img2, max_val=1.0):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(max_val / torch.sqrt(mse))

    def test_attack_adv(self):
        """Adversarial Attack Used Normal Model And Adversarial Model."""

        # Load the trained generator.11111
        # self.restore_model(self.test_iters)
        self.restore_adv_model(self.test_iters)

        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Initialize Metrics
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

        patch_l1_error = 0.0
        patch_l2_error = 0.0
        patch_l1_amp_error =0.0
        patch_l2_amp_error = 0.0
        patch_l1_ang_error = 0.0
        patch_l2_ang_error = 0.0

        for i, (x_real, c_org) in enumerate(data_loader):#外层循环，遍历照片
            # Prepare input images and target domain labels.
            x_real = x_real.to(self.device)
            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

            pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)
            pgd_adv_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None, adv_model=self.adv_G)


            pgd_adv_amp_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None, adv_model=self.adv_G,fft_mode='amp',fft_center=254,fft_protect=False)
            pgd_adv_ang_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None, adv_model=self.adv_G,fft_mode='ang',fft_center=254,fft_protect=False)

            # pgd_adv_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)
            # Translated images.

            x_fake_list = [x_real]

            Add_adv = 0
            Add_noise = 0

            for idx, c_trg in enumerate(c_trg_list):#内层循环，遍历属性
                print('image', i, 'class', idx)
                with torch.no_grad():
                    x_real_mod = x_real
                    # x_real_mod = self.blur_tensor(x_real_mod) # use blur
                    gen_noattack, gen_noattack_feats = self.G(x_real_mod, c_trg)#生成正确伪造
                    adv_gen_noattack, adv_gen_noattack_feats = self.adv_G(x_real_mod, c_trg)#用鲁棒模型生成正确伪造

                # Attacks
                # x_adv, perturb_nom = pgd_attack.perturb(x_real, gen_noattack, c_trg)#普通模型生成对抗样本                     # Vanilla attack
                # x_adv, perturb_adv = pgd_adv_attack.perturb_combine(x_real, gen_noattack, c_trg)#鲁棒模型和普通模型联合生成对抗样本

                # x_adv, perturb = pgd_attack.perturb(x_real, gen_noattack, c_trg)

                x_adv,amp_perturb = pgd_adv_amp_attack.perturb(x_real,gen_noattack,c_trg)
                x_adv,ang_perturb = pgd_adv_ang_attack.perturb(x_real,gen_noattack,c_trg)

                ang_perturb = ang_perturb.rot90(k=2,dims=[1,2])

                #********************还原为图像噪声再相加**************************************************************************
                amp_tensity = 1
                ang_tensity = 0

                # 原始对抗样本
                perturb_total = amp_tensity * amp_perturb + ang_tensity * ang_perturb

                perturb_total = self.perturb_with_luminance(x_real, perturb_total)
                x_adv = x_real + perturb_total

                x_adv_amp,x_adv_ang = self.get_amp_ang(x_adv)
                x_real_amp,x_real_ang = self.get_amp_ang(x_real)


                #************************************************************************************************************************


                #******************振幅噪声直接相加再还原图像********************************************************************************************
                # x_real_amp,x_real_ang = self.get_amp_ang(x_real)
                # amp_tensity = 0.01

                # ang_tensity = 0

                # amp_perturb_total = amp_tensity * amp_perturb  #ang_tensity * ang_perturb 
                # ang_perturb_total = ang_tensity * ang_perturb

                # perturb_total = amp_perturb_total + ang_perturb_total

                # x_adv = (self.recover_fft(x_real_amp + amp_perturb_total ,x_real_ang + ang_perturb_total,x_real,0))

                # x_adv_amp,x_adv_ang = self.get_amp_ang(x_adv)


                #********************************************************************************************************************************

                # No attack
                # x_adv = x_real

                # x_adv = self.blur_tensor(x_adv)   # use blur

                # Metrics
                # #图片质量ssim
                # ssim_values = []
                # ssim_value = ssim(x_real, x_adv, data_range=1.0)
                # print("SSIM:", ssim_value.item())
                # ssim_values.append(ssim_value.item())

                # #PSNR
                # psnr_values = []
                # psnr_value = self.psnr(x_real, x_adv)
                # print("PSNR:", psnr_value.item())
                # psnr_values.append(psnr_value.item())

                with torch.no_grad():
                    gen, _ = self.adv_G(x_adv, c_trg)#生成对抗后伪造输出

                    # Add to lists
                    if Add_adv == 0 :
                        x_fake_list.append(x_adv)
                        Add_adv = 1

                    if Add_noise == 0:
                        x_fake_list.append(perturb_total.reshape(1,3,256,256))
                        Add_noise = 1

                    # x_fake_list.append(blurred_image)
                    #x_fake_list.append(x_adv)
                    #x_fake_list.append(perturb)
                    x_fake_list.append(gen)

                    l1_error += F.l1_loss(gen, gen_noattack)
                    l2_error += F.mse_loss(gen, gen_noattack)
                    l0_error += (gen - gen_noattack).norm(0)
                    min_dist += (gen - gen_noattack).norm(float('-inf'))#计算各项损失参数

                    patch_l1_error += F.l1_loss(x_real,x_adv)
                    patch_l2_error += F.mse_loss(x_real,x_adv)

                    patch_l1_amp_error += F.l1_loss(x_real_amp,x_adv_amp)
                    patch_l2_amp_error += F.mse_loss(x_real_amp,x_adv_amp)

                    patch_l1_ang_error += F.l1_loss(x_real_ang,x_adv_ang)
                    patch_l2_ang_error += F.mse_loss(x_real_ang,x_adv_ang)

                    if F.mse_loss(gen, gen_noattack) > 0.05:
                        n_dist += 1
                    n_samples += 1

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            if i == 49:     # stop after this many images
                break

        # Print metrics
        print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'
              .format(n_samples,l1_error / n_samples, l2_error / n_samples,float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))
        print('adv_error: patch_L1 error: {}. patch_L2 error: {}. patch_L1_amp_error: {}. patch_L2_amp_error: {}. '
              'patch_L1_ang_error: {}. patch_L2_ang_error: {}.'
              .format(patch_l1_error / n_samples,patch_l2_error / n_samples, patch_l1_amp_error / n_samples, patch_l2_amp_error / n_samples, patch_l1_ang_error / n_samples, patch_l2_ang_error / n_samples))
        print('Ne:{}.'.format(l2_error / patch_l2_error))
        # print('Average SSIM: {}. Average PSNR: {}. '.format(np.mean(ssim_values), np.mean(psnr_values)))


    def test_attack_deepfool(self):
        """Test Deepfool adversarial attack"""

        # Load the trained generator.
        self.restore_model(self.test_iters)
        self.restore_adv_model(self.test_iters)

        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Initialize Metrics
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

        for i, (x_real, c_org) in enumerate(data_loader):#外层循环，遍历照片
            # Prepare input images and target domain labels.
            x_real = x_real.to(self.device)
            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

            pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)

            # Translated images.
            x_fake_list = [x_real]

            for idx, c_trg in enumerate(c_trg_list):#内层循环，遍历属性
                print('image', i, 'class', idx)
                with torch.no_grad():
                    x_real_mod = x_real
                    # x_real_mod = self.blur_tensor(x_real_mod) # use blur
                    gen_noattack, gen_noattack_feats = self.G(x_real_mod, c_trg)#生成正确伪造

                # Attacks
                x_adv, perturb = pgd_attack.perturb_deepfool2(x_real, gen_noattack, c_trg)


                perturb = perturb
                # Generate adversarial example
                x_adv = x_real + perturb#生成对抗后图片

                # No attack
                # x_adv = x_real

                # x_adv = self.blur_tensor(x_adv)   # use blur

                # Metrics
                with torch.no_grad():
                    gen, _ = self.G(x_adv, c_trg)#生成对抗后伪造输出

                    # Add to lists
                    # x_fake_list.append(blurred_image)
                    x_fake_list.append(x_adv)
                    # x_fake_list.append(perturb)
                    x_fake_list.append(gen)

                    l1_error += F.l1_loss(gen, gen_noattack)
                    l2_error += F.mse_loss(gen, gen_noattack)
                    l0_error += (gen - gen_noattack).norm(0)
                    min_dist += (gen - gen_noattack).norm(float('-inf'))#计算各项损失参数
                    if F.mse_loss(gen, gen_noattack) > 0.05:
                        n_dist += 1
                    n_samples += 1

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            if i == 49:     # stop after this many images
                break

        # Print metrics
        print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples,
        l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

    def test_attack_nullifying(self):
        """Adversarial Nullifying Attack Used Normal Model And Adversarial Model."""

        # Load the trained generator.
        self.restore_model(self.test_iters)
        self.restore_adv_model(self.test_iters)

        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Initialize Metrics
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

        for i, (x_real, c_org) in enumerate(data_loader):#外层循环，遍历照片
            # Prepare input images and target domain labels.
            x_real = x_real.to(self.device)
            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

            pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)
            pgd_adv_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None, adv_model=self.adv_G)

            # Translated images.
            x_fake_list = [x_real]
            Add_adv=0

            for idx, c_trg in enumerate(c_trg_list):#内层循环，遍历属性
                print('image', i, 'class', idx)
                with torch.no_grad():
                    x_real_mod = x_real
                    # x_real_mod = self.blur_tensor(x_real_mod) # use blur
                    gen_noattack, gen_noattack_feats = self.G(x_real_mod, c_trg)#生成正确伪造
                    adv_gen_noattack, adv_gen_noattack_feats = self.adv_G(x_real_mod, c_trg)#用鲁棒模型生成正确伪造

                # Attacks
                #x_adv, perturb_nom = pgd_attack.perturb(x_real, gen_noattack, c_trg)#普通模型生成对抗样本                     # Vanilla attack
                x_adv, perturb_adv = pgd_adv_attack.perturb_combine_nullifying(x_real, gen_noattack, c_trg)#鲁棒模型和普通模型联合生成对抗样本
                perturb = perturb_adv
                # Generate adversarial example
                x_adv = x_real + perturb#生成对抗后图片

                # No attack
                # x_adv = x_real

                # x_adv = self.blur_tensor(x_adv)   # use blur

                # Metrics
                with torch.no_grad():
                    gen, _ = self.G(x_adv, c_trg)#生成对抗后伪造输出

                    # Add to lists
                    if Add_adv == 0 :
                        x_fake_list.append(x_adv)
                        Add_adv = 1
                    # x_fake_list.append(blurred_image)
                    #x_fake_list.append(x_adv)
                    #x_fake_list.append(perturb)
                    x_fake_list.append(gen)

                    l1_error += F.l1_loss(gen, x_real_mod)
                    l2_error += F.mse_loss(gen, x_real_mod)
                    l0_error += (gen - x_real_mod).norm(0)
                    min_dist += (gen - x_real_mod).norm(float('-inf'))#计算各项损失参数
                    if F.mse_loss(gen, x_real_mod) < 0.05:
                        n_dist += 1
                    n_samples += 1

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            if i == 49:     # stop after this many images
                break

        # Print metrics
        print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples,
        l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))