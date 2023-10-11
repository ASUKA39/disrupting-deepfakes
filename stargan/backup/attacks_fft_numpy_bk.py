import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn
from torch.autograd import Variable
import defenses.smoothing as smoothing

class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.07, k=10, a=0.01, feat = None, adv_model = None,fft_mode = None,fft_center=30,fft_protect=False):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        """
        self.model = model
        self.adv_model = adv_model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

        self.fft_mode = fft_mode
        self.fft_center = fft_center
        self.fft_protect = fft_protect
        # Feature-level attack? Which layer?
        self.feat = feat

        # PGD or I-FGSM?
        self.rand = True

    # 三通道傅立叶变换
    def fft(self, channels):
        '''
        channles: (1,3,256,256)的tensor数组
        return: (3,256,256)的numpy数组
        '''
        
        channels_ = [channels[0][0].cpu().numpy(), channels[0][1].cpu().numpy(), channels[0][2].cpu().numpy()]
        fchannels = []
        for i in range(3):
            channel = np.fft.fft2(channels_[i])
            fchannels.append(np.fft.fftshift(channel))
        return np.array(fchannels)

    # 三通道得到振幅和相位
    def get_amp_ang(self, X):
        '''
        X: (3,256,256)的numpy数组
        return: 2*(1,3,256,256)的tensor.cuda数组
        '''
        
        _fft = self.fft(X)
        amp = []
        cang = []
        for i in range(3):
            amp.append(np.abs(_fft[i]))
            cang.append(np.angle(_fft[i]))

        _amp = []
        _cang =[]
        _amp.append((torch.DoubleTensor(torch.from_numpy(amp[0]))).cuda())
        _amp.append((torch.DoubleTensor(torch.from_numpy(amp[1]))).cuda())
        _amp.append((torch.DoubleTensor(torch.from_numpy(amp[2]))).cuda())

        _cang.append((torch.DoubleTensor(torch.from_numpy(cang[0]))).cuda())
        _cang.append((torch.DoubleTensor(torch.from_numpy(cang[1]))).cuda())
        _cang.append((torch.DoubleTensor(torch.from_numpy(cang[2]))).cuda())

        _amp = torch.stack((_amp[0], _amp[1], _amp[2]), 0)
        _amp = torch.unsqueeze(_amp,0)
        _amp = _amp.to(torch.float64)

        _cang = torch.stack((_cang[0], _cang[1], _cang[2]), 0)
        _cang = torch.unsqueeze(_cang,0)
        _cang = _cang.to(torch.float64)

        return _amp, _cang


    def protect(slef,now,backup,_range):
        rows, cols = now[0][0].shape

        row, col = rows // 2, cols // 2

        for i in range(3):
            now[0][i][row - _range:row + _range, col - _range:col + _range] = backup[0][i][row - _range:row + _range,
                                                                            col - _range:col + _range]
            
        return now
    
    # 加噪复原
    def recover(self, amp, ang,backup):
        '''
        amp,ang:两个(1,3,256,256)tensor.cuda数组
        return: (1,3,256,256)tensor
        '''
        if self.fft_protect == True :
            if self.fft_mode == 'amp':
                amp = self.protect(amp,backup,self.fft_center)

            elif self.fft_mode == 'ang':
                ang = self.protect(ang,backup,self.fft_center)

        _re = []
        for i in range(3):
            re = (amp[0][i].cpu().numpy() * np.cos(ang[0][i].cpu().numpy()) + (amp[0][i].cpu().numpy() )* np.sin(ang[0][i].cpu().numpy()) * 1j)
            re = np.fft.ifftshift(re)
            re = np.fft.ifft2(re)
            re = np.abs(re)
            _re.append(np.uint8(re))
        #torch.tensor(X)).to(torch.float32).cuda(), (torch.tensor(per * 100)).to(torch.float32).cuda()
        _re = np.array(_re)
        return torch.tensor(_re).to(torch.float32).cuda()

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
           
            
            #[numpy,numpy,numpy]
            X_amp, X_ang = self.get_amp_ang(X)

            Y_amp, Y_ang = self.get_amp_ang(y)

            if self.fft_mode == 'amp':
                X = X_amp
                Y = Y_amp
                X_backup = X.clone().detach()
            
            elif self.fft_mode == 'ang':
                X = X_ang
                Y = Y_ang
                X_backup = X.clone().detach()

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

        if self.fft_mode == 'amp':
            per = self.recover(X -X_backup,X_ang,X_backup)
            X = self.recover(X, X_ang,X_backup)
            return X ,per
        elif self.fft_mode == 'ang':
            per = self.recover(X_amp,X -X_backup,X_backup)
            X = self.recover(X_amp,X,X_backup)
            return X,per
        
        # return X, (per * 100)
        return X, X - X_nat

    def perturb_blur(self, X_nat, y, c_trg):
        """
        White-box attack against blur pre-processing.
        """
        if self.rand:#PGD
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
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
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
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
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
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
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
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
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
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
    
    def perturb_combine(self, X_nat, y, c_trg):
        """
        Normal Model And Adversarial Model Combine Attack.
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            #X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()

        for i in range(self.k):
            X.requires_grad = True
            output_nor, feats_nor = self.model(X, c_trg)
            output_adv, feats_adv = self.adv_model(X, c_trg)

            #output = (0.5 * output_nor) + (1 * output_adv)

            self.model.zero_grad()
            # Minus in the loss means "towards" and plus means "away from"
            #loss = self.loss_fn(output_nor, X_nat) + self.loss_fn(output_adv, X_nat)#y是正常伪造结果，output是加了样本后伪造结果，计算二者间损失
            #loss = self.loss_fn(output, y)
            loss = 0*self.loss_fn(output_nor, y)+self.loss_fn(output_adv, y)
            loss.backward()#反向传播
            grad = X.grad#计算梯度

            X_adv = X + self.a * grad.sign()#添加样本

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat
    
    def perturb_deepfool(self, X_nat, y, c_trg):
        """
        Deepfool Attack.
        """
        max_iter=1
        overshoot=0.001

        
        X = X_nat.clone().detach_()
        X.requires_grad = True
        iter = 0
        pert=torch.zeros_like(X_nat)
        r_tot =torch.zeros_like(X_nat)
        while iter < max_iter:
            output, feats = self.model(X, c_trg)
            self.model.zero_grad()
            loss=self.loss_fn(output, y)
            loss.backward(retain_graph=True)
            grad=X.grad
            pert=abs(loss) / grad.norm() * grad / grad.norm()
            #r_i=(1+overshoot)*pert
            #r_tot=r_tot+r_i
            X=X+pert
            iter=iter+1
        
        #r_tot = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
        #X = torch.clamp(X_nat + r_tot, min=-1, max=1).detach_()
        #X = torch.clamp(X_nat, min=-1, max=1).detach_()
        self.model.zero_grad()

        return X, X-X_nat

    def perturb_deepfool2(self, X_nat, y, c_trg):
        """
        使用DeepFool算法生成对抗样本
        参数：
        - X_nat: 原始输入样本
        - y: 原始输入样本的标签
        - c_trg: 目标类别的索引
        返回值：
        - X_adv: 生成的对抗样本
        """
        # 将输入样本转换为张量
        X_nat = torch.Tensor(X_nat).float()
        y = torch.Tensor(y).float()

        # 将模型设置为评估模式
        self.model.eval()

        # 将输入样本包装为变量
        X_nat_var = Variable(X_nat, requires_grad=True)

        # 获取输入样本的梯度
        output, feats = self.model(X_nat_var, c_trg)
        loss = torch.nn.functional.cross_entropy(output, y)
        self.model.zero_grad()
        loss.backward()
        grad = X_nat_var.grad.data

        # 使用DeepFool算法生成对抗样本
        X_adv = X_nat.clone()
        perturbation = np.zeros_like(X_nat.cpu())

        for i in range(len(X_nat)):
            x = X_nat_var[i].unsqueeze(0)
            x.requires_grad = True
            fs = self.model(x)
            k_i = torch.argmax(fs).item()

            w = torch.zeros_like(x)
            r_tot = torch.zeros_like(x)

            while k_i == y[i]:
                pert = np.inf
                fs[0, y[i]].backward(retain_graph=True)
                grad_orig = x.grad.data.clone()

                for k in range(len(fs[0])):
                    if k == y[i]:
                        continue

                    x.grad.data.zero_()
                    fs[0, k].backward(retain_graph=True)
                    cur_grad = x.grad.data.clone()

                    w_k = cur_grad - grad_orig
                    f_k = fs[0, k] - fs[0, y[i]]

                    pert_k = abs(f_k.item()) / torch.norm(w_k.flatten())

                    if pert_k < pert:
                        pert = pert_k
                        w = w_k

                r_i = (pert + 1e-4) * w / torch.norm(w)
                r_tot += r_i

                x.grad.data.zero_()
                fs = self.model(x + r_tot)
                k_i = torch.argmax(fs).item()

            perturbation[i] = r_tot.squeeze().detach().numpy()
            X_adv[i] = (X_nat[i] + perturbation[i]).clip(0, 1)

        return X_adv, X_adv-X_nat
    
    def perturb_combine_nullifying(self, X_nat, y, c_trg):
        '''
        Normal Model And Adversarial Model Combine Nullifying Attack.
        '''
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            #X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()

        for i in range(self.k):
            X.requires_grad = True
            output_nor, feats_nor = self.model(X, c_trg)
            output_adv, feats_adv = self.adv_model(X, c_trg)

            #output = (0.5 * output_nor) + (1 * output_adv)

            self.model.zero_grad()
            # Minus in the loss means "towards" and plus means "away from"
            #loss = self.loss_fn(output_nor, X_nat) + self.loss_fn(output_adv, X_nat)#y是正常伪造结果，output是加了样本后伪造结果，计算二者间损失
            #loss = self.loss_fn(output, y)
            loss = 0.5*self.loss_fn(output_nor, y)+self.loss_fn(output_adv, y)
            loss.backward()#反向传播
            grad = X.grad#计算梯度

            X_adv = X - self.a * grad.sign()#添加样本

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat


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