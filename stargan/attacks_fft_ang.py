import copy
import numpy as np
# from collections import Iterable
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

import cv2

import defenses.smoothing as smoothing


class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=10, a=0.01, feat=None):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

        # Feature-level attack? Which layer?
        self.feat = feat

        # PGD or I-FGSM?
        self.rand = True

    # 三通道傅立叶变换
    def fft(self, channels):
        fchannels = []
        for i in range(3):
            channel = np.fft.fft2(channels[i])
            fchannels.append(np.fft.fftshift(channel))
        return fchannels

    # 三通道得到振幅和相位
    def map_ang(self, _fft):
        cmap = []
        cang = []
        for i in range(3):
            cmap.append(np.abs(_fft[i]))
            cang.append(np.angle(_fft[i]))
        return cmap, cang

    # 振幅加噪
    def map_noise(self, _noise, cmap, tensity, _range):
        rows, cols = cmap[0].shape

        row, col = rows // 2, cols // 2

        mask = np.zeros((rows, cols))
        mask[row - 30:row + 30, col - 30:col + 30] = 1

        temp_bgr = []
        for i in range(3):
            temp_bgr.append(cmap[i] * mask)
            cmap[i] += _noise * tensity
            cmap[i][row - _range:row + _range, col - _range:col + _range] = temp_bgr[i][row - _range:row + _range,
                                                                            col - _range:col + _range]

        return cmap

    # 相位加噪
    def ang_noise(self, _noise, cang, tensity, _range):
        rows, cols = cang[0].shape

        row, col = rows // 2, cols // 2

        mask = np.zeros((rows, cols))
        mask[row - 30:row + 30, col - 30:col + 30] = 1

        temp_bgr = []
        for i in range(3):
            temp_bgr = cang[i] * mask
            cang[i] += _noise * tensity
            cang[i][row - _range:row + _range, col - _range:col + _range] = temp_bgr[row - _range:row + _range,
                                                                            col - _range:col + _range]

        return cang

    # 加噪复原
    def reverse(self, cmap, ang):
        _re = []
        for i in range(3):
            # 
            re = (cmap[i] * np.cos(ang[i].cpu().numpy()) + cmap[i] * np.sin(ang[i].cpu().numpy()) * 1j)
            re = np.fft.ifftshift(re)
            re = np.fft.ifft2(re)
            re = np.abs(re)
            _re.append(np.uint8(re))
        return np.abs(_re)

    def perturb(self, X_nat, y, c_trg):
        """
        Vanilla Attack.
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(
                np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()    

        
        for i in range(self.k):
            # 将输入的图像转为幅度向量，在幅度向量上计算
            X_rgb = [X[0][0].cpu().numpy(), X[0][1].cpu().numpy(), X[0][2].cpu().numpy()]
            fft_rgb = self.fft(X_rgb)
            X_map, X_ang = self.map_ang(fft_rgb)

            X_ang[0] = (torch.DoubleTensor(torch.from_numpy(X_ang[0]))).cuda()
            X_ang[1] = (torch.DoubleTensor(torch.from_numpy(X_ang[1]))).cuda()
            X_ang[2] = (torch.DoubleTensor(torch.from_numpy(X_ang[2]))).cuda()

            X = torch.stack((X_ang[0], X_ang[1], X_ang[2]), 0)
            X = torch.unsqueeze(X, 0)
            X = X.to(torch.float64)

            Y = y
            Y_rgb = [Y[0][0].cpu().numpy(), Y[0][1].cpu().numpy(), Y[0][2].cpu().numpy()]
            fft_rgb = self.fft(Y_rgb)
            Y_map, Y_ang = self.map_ang(fft_rgb)

            Y_ang[0] = (torch.DoubleTensor(torch.from_numpy(Y_ang[0]))).cuda()
            Y_ang[1] = (torch.DoubleTensor(torch.from_numpy(Y_ang[1]))).cuda()
            Y_ang[2] = (torch.DoubleTensor(torch.from_numpy(Y_ang[2]))).cuda()

            Y = torch.stack((Y_ang[0], Y_ang[1], Y_ang[2]), 0)
            Y = torch.unsqueeze(Y, 0)
            Y = Y.to(torch.float64)

            X.requires_grad = True

            # 在幅度向量上计算生成
            output, feats = self.model(X.float(), c_trg)

            if self.feat:
                output = feats[self.feat]

            self.model.zero_grad()
            # Minus in the loss means "towards" and plus means "away from"

            loss = self.loss_fn(output.float(), Y.float())

            loss = loss.float()
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        X_nat_rgb = [X_nat[0][0].cpu().numpy(), X_nat[0][1].cpu().numpy(), X_nat[0][2].cpu().numpy()]
        fft_rgb = self.fft(X_nat_rgb)
        X_nat_map, X_nat_ang = self.map_ang(fft_rgb)

        X_nat_ang[0] = (torch.DoubleTensor(torch.from_numpy(X_nat_ang[0]))).cuda()
        X_nat_ang[1] = (torch.DoubleTensor(torch.from_numpy(X_nat_ang[1]))).cuda()
        X_nat_ang[2] = (torch.DoubleTensor(torch.from_numpy(X_nat_ang[2]))).cuda()

        X_nat = torch.stack((X_nat_ang[0], X_nat_ang[1], X_nat_ang[2]), 0)
        X_nat = torch.unsqueeze(X_nat, 0)
        X_nat = X_nat.to(torch.float64)

        X_nat = X_nat.reshape(3, 256, 256)
        X = X.reshape(3, 256, 256)

        per = self.reverse(X_map, X - X_nat)
        X = self.reverse(X_map, X)
        
        X = X.reshape(1, 3, 256, 256)
        per = per.reshape(1,3,256,256)
        # return X, (per * 100)
        strg = 0.3
        return (torch.tensor(X)).to(torch.float32).cuda(), (torch.tensor(per * strg)).to(torch.float32).cuda()

    def perturb_blur(self, X_nat, y, c_trg):
        """
        White-box attack against blur pre-processing.
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(
                np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()  

        X_orig = X_nat.clone().detach_()

        # Kernel size
        ks = 11
        # Sigma for Gaussian noise
        sig = 1.5

        # preproc = smoothing.AverageSmoothing2D(channels=3, kernel_size=ks).to(self.device)
        preproc = smoothing.GaussianSmoothing2D(sigma=sig, channels=3, kernel_size=ks).to(self.device)

        # blurred_image = smoothing.AverageSmoothing2D(channels=3, kernel_size=ks).to(self.device)(X_orig)
        blurred_image = smoothing.GaussianSmoothing2D(sigma=sig, channels=3, kernel_size=ks).to(self.device)(X_orig)

        for i in range(self.k):
            X.requires_grad = True
            output, feats = self.model.forward_blur(X, c_trg, preproc)

            self.model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat, blurred_image

    def perturb_blur_iter_full(self, X_nat, y, c_trg):
        """
        Spread-spectrum attack against blur defenses (gray-box scenario).
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(
                np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()  

        # Gaussian blur kernel size
        ks_gauss = 11
        # Average smoothing kernel size
        ks_avg = 3
        # Sigma for Gaussian blur
        sig = 1
        # Type of blur
        blur_type = 1

        for i in range(self.k):
            # Declare smoothing layer
            if blur_type == 1:
                preproc = smoothing.GaussianSmoothing2D(sigma=sig, channels=3, kernel_size=ks_gauss).to(self.device)
            elif blur_type == 2:
                preproc = smoothing.AverageSmoothing2D(channels=3, kernel_size=ks_avg).to(self.device)

            X.requires_grad = True
            output, feats = self.model.forward_blur(X, c_trg, preproc)

            if self.feat:
                output = feats[self.feat]

            self.model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

            # Iterate through blur types
            if blur_type == 1:
                sig += 0.5
                if sig >= 3.2:
                    blur_type = 2
                    sig = 1
            if blur_type == 2:
                ks_avg += 2
                if ks_avg >= 11:
                    blur_type = 1
                    ks_avg = 3

        self.model.zero_grad()

        return X, X - X_nat

    def perturb_blur_eot(self, X_nat, y, c_trg):
        """
        EoT adaptation to the blur transformation.
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(
                np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()  

        # Gaussian blur kernel size
        ks_gauss = 11
        # Average smoothing kernel size
        ks_avg = 3
        # Sigma for Gaussian blur
        sig = 1
        # Type of blur
        blur_type = 1

        for i in range(self.k):
            full_loss = 0.0
            X.requires_grad = True
            self.model.zero_grad()

            for j in range(9):  # 9 types of blur
                # Declare smoothing layer
                if blur_type == 1:
                    preproc = smoothing.GaussianSmoothing2D(sigma=sig, channels=3, kernel_size=ks_gauss).to(self.device)
                elif blur_type == 2:
                    preproc = smoothing.AverageSmoothing2D(channels=3, kernel_size=ks_avg).to(self.device)

                output, feats = self.model.forward_blur(X, c_trg, preproc)

                loss = self.loss_fn(output, y)
                full_loss += loss

                if blur_type == 1:
                    sig += 0.5
                    if sig >= 3.2:
                        blur_type = 2
                        sig = 1
                if blur_type == 2:
                    ks_avg += 2
                    if ks_avg >= 11:
                        blur_type = 1
                        ks_avg = 3

            full_loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat

    def perturb_iter_class(self, X_nat, y, c_trg):
        """
        Iterative Class Conditional Attack
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(
                np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()  

        j = 0
        J = len(c_trg)

        for i in range(self.k):
            X.requires_grad = True
            output, feats = self.model(X, c_trg[j])

            self.model.zero_grad()

            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

            j += 1
            if j == J:
                j = 0

        return X, eta

    def perturb_joint_class(self, X_nat, y, c_trg):
        """
        Joint Class Conditional Attack
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(
                np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()  

        J = len(c_trg)

        for i in range(self.k):
            full_loss = 0.0
            X.requires_grad = True
            self.model.zero_grad()

            for j in range(J):
                output, feats = self.model(X, c_trg[j])

                loss = self.loss_fn(output, y)
                full_loss += loss

            full_loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        return X, eta


def clip_tensor(X, Y, Z):
    # Clip X with Y min and Z max
    X_np = X.data.cpu().numpy()
    Y_np = Y.data.cpu().numpy()
    Z_np = Z.data.cpu().numpy()
    X_clipped = np.clip(X_np, Y_np, Z_np)
    X_res = torch.FloatTensor(X_clipped)
    return X_res


def perturb_batch(X, y, c_trg, model, adversary):
    # Perturb batch function for adversarial training
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()

    adversary.model = model_cp

    X_adv, _ = adversary.perturb(X, y, c_trg)

    return X_adv
