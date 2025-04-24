# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:09:32 2025

@author: rmiele
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import torch.nn.functional as F
from tqdm import tqdm
import time

class DPS_Inversion():
    def __init__(self, args):
        
        #load network
        with open(args.net_dir+args.net_snapfile, 'rb') as f:
            self.net = pickle.load(f)['ema'].to(args.device)
            
        self.device = args.device
        self.sigma_max = args.sigma_max
        self.sigma_min = 0.002
        self.image_size = args.image_size
        self.N = torch.prod(torch.tensor(self.image_size)).to(self.device)
        self.image_depth = args.image_depth
        self.rho = args.rho
        self.n_samples = args.n_samples
        self.workdir = args.workdir
        self.num_steps = args.num_steps
        self.save_dir = args.save_dir
        self.hard_data = args.hard_data
        self.seismic = args.seismic
        self.plot_progress = False
        
        #loading test data---------------------------------------------------------------
        self.true_Ip= torch.load(args.test_models_dir+f'/mtest_ip{args.test_model_n}.pt', 
                                 weights_only=True, map_location=self.device)[None,None,:].double()
        self.true_F= torch.load(args.test_models_dir+f'/mtest_facies{args.test_model_n}.pt', 
                                 weights_only=True, map_location=self.device)[None,None,:].double()
        self.ip= [self.true_Ip.min(),self.true_Ip.max(), self.true_Ip.max()-self.true_Ip.min()]
        
        if self.image_size[-1]==100: 
            self.true_Ip= self.pad(self.true_Ip)
            self.true_F= self.pad(self.true_F)
            
        #conditioning on seismic data----------------------------------------------------
        if self.seismic: 
            self.load_wavelet_define_conv(args)
            self.ys_obs = self.physics_forward(self.true_Ip)
            torch.save(self.ys_obs, self.save_dir+'/ys_obs.pt')
            
            self.conv_jacobian = self.conv_matrix_J().to(self.device)
            self.len_wav = len(self.wavelet.squeeze())
        
            #assumed noise vector
            self.sigma_ys = args.relative_data_error*torch.abs(self.ys_obs.detach().clone())+args.absolute_data_error
            self.var_ys = self.sigma_ys**2
        
            #contaminate with noise
            self.noise_ys = torch.randn_like(self.ys_obs)*self.sigma_ys
            self.ys_obs = (self.ys_obs + self.noise_ys)
            self.yslims= [self.ys_obs.min(),self.ys_obs.max()]
            torch.save(self.noise_ys, self.save_dir+'/noise_ys.pt')
            if self.image_size[-1]==100: 
                self.ys_obs[...,100:] = 0
                self.sigma_ys[...,100:] = 0
                self.var_ys[...,100:] = 0
            
            
        #conditioning on hard data-------------------------------------------------------
        if self.hard_data:
            self.mask = torch.zeros((1,self.image_depth,
                                      self.image_size[0],
                                      self.image_size[1])).to(self.device)
            if self.image_size[-1]==100: 
                self.mask = self.pad(self.mask)
            
            for i in args.hd_cond_where:
                if i==30:
                    self.mask[...,:40,:100] = torch.flip(torch.diag(torch.ones(100),15)[...,:40,:100],
                                                         [1])
                else:
                    self.mask[...,i] = 1
            
            
            self.yh_obs = self.mask*self.true_F.squeeze()
            self.yh_obs[:,1] = self.mask[:,1]*((self.true_Ip-self.ip[0])/self.ip[2]).squeeze()
            
            self.sigma_yh = torch.zeros_like(self.yh_obs).to(self.device)
            args.hard_data_error = np.array(args.hard_data_error)
            self.sigma_yh[:,0] = args.hard_data_error[0]
            self.sigma_yh[:,1] = args.hard_data_error[1]
            torch.save(self.yh_obs, self.save_dir+'/yh_obs.pt')
            #contaminate with noise
            self.noise_yh = torch.randn_like(self.yh_obs)*self.sigma_yh*self.mask
            self.yh_obs+=self.noise_yh
            self.N_rmse = torch.sum(self.mask)
            
            torch.save(self.yh_obs, self.save_dir+'/noise_yh.pt')
            fig, axs = plt.subplots(1,2)
            ax = axs[0].imshow(self.yh_obs[0,0].detach().cpu())
            plt.colorbar(ax)
            ax= axs[1].imshow(self.yh_obs[0,1].detach().cpu())
            plt.colorbar(ax)
            plt.show()
            
    def pad (self,array):
        return F.pad(array, (0, 4, 0, 0))

        
    def compute_estimation_error(self, test_data):
        # define the estimation errors at each timestep, individually from a set of test data
        try: 
            self.sigma_xhat0 = torch.load(self.workdir+f'/xhat0_sigma_{self.sigma_max}_{self.rho}_{self.num_steps}.pt', weights_only=True, map_location=self.device)
        except: 
            N_ex = test_data.shape[0]
            div = 4 if N_ex > 4 else 1
            bs = int(N_ex/div) #batch size
            self.sigma_xhat0=torch.zeros(self.num_steps+1,self.image_depth).to(self.device)
            
            # define the sigma sampling schedule 
            # Adjust noise levels based on what's supported by the network.
            sigma_min = max(self.sigma_min, self.net.sigma_min)
            sigma_max = min(self.sigma_max, self.net.sigma_max)
            
            # Time step discretization.
            step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=self.device)
            t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
            
            for i in tqdm(range(len(t_steps)), desc= 'Error at time-step'):
                #error vectors
                err_y = torch.randn_like(test_data).to(self.device)*t_steps[i]
                residuals = torch.zeros_like(test_data)
                #estimate x_0 at specific error
                D_yn = torch.zeros_like(test_data)
                for j in range(div):
                    D_yn[j*bs:j*bs+bs] = self.net(test_data[j*bs:j*bs+bs] + err_y[j*bs:j*bs+bs], t_steps[i], None)
                    residuals[j*bs:j*bs+bs] = (test_data[j*bs:j*bs+bs]-D_yn[j*bs:j*bs+bs])
                if j*bs+bs != N_ex:
                    D_yn[j*bs+bs:] = self.net(test_data[j*bs+bs:] + err_y[j*bs+bs:], t_steps[i], None)
                    residuals[j*bs+bs:] = (test_data[j*bs+bs:]-D_yn[j*bs+bs:])
    
                #compute the residual for different images at noise level t
                mean_err_pixel = torch.sqrt(torch.mean((residuals.flatten(2)/2)**2, dim=0))
                self.sigma_xhat0[i,:] = torch.quantile(mean_err_pixel.flatten(1), q=0.5, dim=1)
            torch.save(self.sigma_xhat0, self.workdir+f'/xhat0_sigma_{self.sigma_max}_{self.rho}_{self.num_steps}.pt')

            plt.figure()
            x = np.arange(0,len(t_steps),5)
            plt.title(r'Uncertainty on $\hat{x}_0$ per time-step')
            plt.xticks(x,[t_steps.detach().cpu().numpy()[i].round(1) for i in x])
            plt.plot(np.array(self.sigma_xhat0[:,0].detach().cpu()), label='facies')
            plt.plot(np.array(self.sigma_xhat0[:,1].detach().cpu()), label='Ip')
            plt.legend()
            plt.ylabel(r'$\sigma_{\hat{x}_0} $')
            plt.xlabel(r'$\sigma(t) $')
            plt.savefig(self.save_dir+'/sigma_xhat0.png')
            plt.show()
            plt.close('all')
            del test_data,err_y,residuals
            torch.cuda.empty_cache()
            
        return None


    def load_wavelet_define_conv(self, args):
        # wavelet = np.genfromtxt(args.workdir+args.inv_folder+args.wavelet_file)*args.w_scale
        frequency = 1/(torch.pi*5)
        t= torch.arange(-20,20,1)
        omega = torch.pi * frequency
        wavelet = ((1 - 2 * (omega * t) ** 2) * torch.exp(-(omega * t) ** 2))*100
        
        wavelet = np.expand_dims(wavelet, 0) # add bacth [B x H x W x C]
        if wavelet.ndim==2: 
            wavelet= np.expand_dims(wavelet, 0)
            wavelet= np.expand_dims(wavelet, -1)
            
        self.wavelet = torch.from_numpy(wavelet).double().to(args.device)
        k = self.wavelet.shape[-2]
        self.padding = (k//2,0)
        self.seismic_conv = torch.nn.Conv2d(1,1, kernel_size=1, padding= self.padding, bias=False)
        self.seismic_conv.weight = torch.nn.Parameter(self.wavelet).requires_grad_(False)
        
        return None


    def physics_forward(self, realization):
        ip = torch.cat((realization, realization[:,:,[-1],:]), dim=2) # repeats last element
        ip_d =  ip[:, :, 1:, :] - ip[:, :, :-1, :]
        ip_a = (ip[:, :, 1:, :] + ip[:, :, :-1, :])    
        rc = ip_d / ip_a
        return self.seismic_conv(rc)[:,:,:self.image_size[0],:]


    def diag_els (self, xi, xii):
        #Jacobian of reflectivity coefficients  models
        den = (xi+xii)**2
        num1 = -2*xii
        num2 = 2*xi
        return torch.tensor([num1/den, num2/den])


    def conv_matrix_J(self):
        #creates the matrix for 1D convolution, input data is padded
        #same as torch, but torch is more efficient at forward, this is needed for Jacobian 
        wavelet_lenght = len(self.wavelet.squeeze())
        trace_lenght = self.image_size[0]+wavelet_lenght
        toeplitz_w = torch.zeros(trace_lenght-wavelet_lenght+1-wavelet_lenght%2, 
                                 trace_lenght) 
        
        for i in range(toeplitz_w.shape[0]):
            toeplitz_w[i,i:i+wavelet_lenght]=self.wavelet.squeeze()
            
        return toeplitz_w


    def cond_grad_HD(self, x_0hat, x_t, i=0):
        
        grads = torch.zeros_like(x_0hat.detach())
        wrmse = torch.zeros(self.minibatch)
        covar = torch.zeros_like(self.sigma_yh).to(self.device)
        
        x_0hat_r = (x_0hat+1)/2
        d_hat = x_0hat_r*self.mask
        difference = ((self.yh_obs - d_hat)**2)*self.mask

        if args.error_x0 == 'prop':
            #covariance is the noise in the observed data + the noise in the image itself (both Gaussian uncorrelated so just add diagoanl matrices)
            for p in range(self.image_depth):
                covar[:,p] = self.sigma_yh[:,p]**2 + self.sigma_xhat0[i][p]**2

        elif args.error_x0 == 'dps':
            #covariance is the noise in the observed data 
            covar = self.sigma_yh**2

        
        if args.error_x0 != 'actual_dps':
            #Maximum likelihood for Gaussian Noise (not implemented for other types)
            err = (difference/covar)
            norm = - .5 * torch.sum(err.flatten(1), dim=1)
        
        else:
            #this is the way they actually implement it in Chung et al. 2022 for DPS inversion (not correct)
            norm = - torch.linalg.norm(difference)[None,]
            
        for i in range(self.minibatch): 
            #loop is required for gradient caluclation independent for each sample
            grads[i] = torch.autograd.grad(outputs=norm[i], inputs=x_t, retain_graph=True)[0][i].detach() #gradients        
            wrmse[i] = ((difference/(self.sigma_yh**2)).sum()/self.N_rmse).sqrt().detach().cpu()
            
            torch.cuda.empty_cache()
            
        return grads, wrmse


    def cond_grad_nonlinear(self, x_0hat,x_t,i=0):
        #compute d_hat and error from observed
        grads = torch.zeros_like(x_0hat.detach())
        wrmse = torch.zeros(self.minibatch)
        norm = torch.zeros(self.minibatch)
        
        x_0hat_r = (x_0hat+1)/2
        x_0hat_r_ip = ((x_0hat_r[:,1,None])*self.ip[2]+self.ip[0])
        d_hat = self.physics_forward(x_0hat_r_ip)
        difference = (d_hat - self.ys_obs)
        
        if args.error_x0 == 'prop':
            assert(self.minibatch==1), 'Not implemented for > 1 realization at a time'
            
            #take the sigma value for xhat0 at time t_hat - rescaled to Ip support
            sigma_x0 = self.sigma_xhat0[i][1]*self.ip[2]
        
            #get the corresponding covariance matrix for Gaussian uncorrelated noise (d x d, d=len(seismic trace))
            temp_eye = torch.eye(self.image_size[-2]).double().to(self.device)*(sigma_x0**2)
            
            #Jacobian structure for RC (changes value at each trace)
            JRC_struct = torch.zeros(self.conv_jacobian.shape[1],self.conv_jacobian.shape[1],).to(self.device)
            
            #inverse of the full covariances are saved to maintain autograd stable
            inv_cov = torch.zeros(self.image_size[-1], self.image_size[-2], self.image_size[-2]).to(self.device)
            norm_t = torch.zeros(self.image_size[-1])

            for t in range(self.image_size[1]):
                trace = x_0hat_r_ip.squeeze()[:,t]
                
                # if 2nd is trace[i], assume last element is repeated
                # idx = self.len_wav//2-1
                # JRC_struct[idx,idx:idx+2]= self.diag_els(trace[0], trace[0]) 
                for i in range(self.image_size[0]-1):
                    idx = i+self.len_wav//2
                    JRC_struct[idx,idx:idx+2] = self.diag_els(trace[i], trace[i+1])
                idx = i+1+self.len_wav//2
                JRC_struct[idx,idx]= self.diag_els(trace[i], 0)[0] # if 2nd is trace[i], assume last element is repeated

                J_trace = (self.conv_jacobian @ JRC_struct)[:-1, self.len_wav//2:-self.len_wav//2].double()
            
                #Full covariance matrix = Error propagation + observed data error
                cov = torch.matmul(J_trace.T, torch.matmul(temp_eye, J_trace))
                cov += torch.diag(self.var_ys[0,0,:,t]).to(self.device)
                
                inv_cov[t]  = torch.inverse(cov) 
                norm_t[t] = - .5*torch.matmul(difference[0,0,:,t], torch.matmul(inv_cov[t].double(),  difference[0,0,:,t]))

            norm = torch.sum(norm_t)
            
        else:
            err = ((difference)[...,:100]**2/(self.var_ys)[...,:100])
            norm = - .5* torch.sum(err.flatten(1), dim=1)

        grads = torch.autograd.grad(outputs=norm, inputs=x_t, retain_graph=True)[0].detach()
        
        difference.detach_()
        wrmse = ((((difference/self.sigma_ys)**2)[...,:100]).sum()/self.N).sqrt().detach().cpu()

        torch.cuda.empty_cache()
        
        return grads, wrmse
    
    
    def condition(self, denoised, x_hat, i):
        cond_grads = torch.zeros_like(denoised)
        rmset= torch.zeros(self.minibatch, 2)
        if self.hard_data: 
            gradients, rmse= self.cond_grad_HD(denoised, x_hat, i)      #calculate the gradients and loss
            cond_grads+= gradients
            rmset[:,0]= rmse
            
        if self.seismic: 
            gradients, rmse= self.cond_grad_nonlinear(denoised, x_hat, i)          #calculate the gradients and loss
            cond_grads+= gradients
            rmset[:,1]=rmse
        return cond_grads, rmset
    
        
    def edm_sampler(self, 
                    n_samples, class_labels=None, randn_like=torch.randn_like,
                     S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):  
                     # S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):  
        """
            This is a modified edm sampler to include conditioning to hard data and seismic
        """
        
        self.minibatch = n_samples
        
        if self.seismic: wrmse_s_t = torch.zeros(self.minibatch, self.num_steps)
        if self.hard_data: wrmse_h_t = torch.zeros(self.minibatch, self.num_steps)
        
        latents = torch.randn(self.minibatch,self.image_depth,self.image_size[0],self.image_size[1]).to(self.device)
        
        if self.image_size[1]==100: 
            latents = torch.randn(self.minibatch,self.image_depth,self.image_size[0],self.image_size[1]+4).to(self.device)
        
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        
        # Main sampling loop.
        text = f"{'Seismic' if self.seismic else ''} {'Hard data' if self.hard_data else ''}"
        desc = f'Condintioning on {text} - {self.minibatch} samples, {self.num_steps} steps' if (self.seismic or self.hard_data) else None
        x_next = latents.to(torch.float64) * t_steps[0]
        
        pbar = tqdm(enumerate(zip(t_steps[:-1], t_steps[1:])), desc= desc)
        for i, (t_cur, t_next) in pbar: # 0, ..., N-1
            
            x_cur = x_next
            
            # Increase noise temporarily.
            gamma = min(S_churn / self.num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
            
            #evaluate
            x_hat.requires_grad_(True)
            denoised = self.net(x_hat, t_hat, class_labels).to(torch.float64) #get xhat_0|x_t
            cond_grads, wrmse = self.condition(denoised, x_hat,i) #get physics gradients dL/dx_t
            
            #preserve memory
            denoised.detach_()
            x_hat.detach_()
            x_next.detach_()
            torch.cuda.empty_cache()
            
            #compute next step (Euler)
            score_func = (denoised-x_hat)/(t_hat**2)
            if args.error_x0 == 'actual_dps': #their way, adapted to EDM
                d_cur = - t_hat * (score_func)
                x_next = x_hat + (t_next - t_hat) * d_cur - cond_grads
                
            else:
                d_cur = - t_hat * (score_func + cond_grads)
                x_next = x_hat + (t_next - t_hat) * d_cur
            
            if args.error_x0 != 'actual_dps': #let's just skip this is not necessary
                # Apply 2nd order correction.
                if i < self.num_steps - 1:
                    
                    #evaluate
                    x_next.requires_grad_(True)
                    denoised = self.net(x_next, t_next.unsqueeze(0), class_labels).to(torch.float64) #get xhat_0|x_next
                    cond_grads, wrmse = self.condition(denoised, x_next,i) #get physics gradients dL/dx_next
                    
                    #preserve memory
                    denoised.detach_()
                    x_hat.detach_()
                    x_next.detach_()
                    torch.cuda.empty_cache()
                    
                    #compute next step (2nd order)
                    score_func = (denoised-x_hat)/(t_hat**2)
                    d_prime = - t_hat * (score_func + cond_grads)
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    
            pbar.set_postfix({'WRMSE': wrmse}, refresh=False)
        
            #store rmse for logging progress
            if self.hard_data: wrmse_h_t[:,i] = wrmse[:,0]
            if self.seismic: wrmse_s_t[:,i] = wrmse[:,1]
            if self.plot_progress : self.pltprgr(denoised,cond_grads,i)
        #storing stuff for safety
        try:
            temp_real = torch.load(self.save_dir+'/realizations_temp.pt', weights_only = True, map_location='cpu')
            if self.hard_data: temp_wrmse_h = torch.load(self.save_dir+'/wrmse_h_temp.pt', weights_only = True, map_location='cpu')
            if self.seismic: temp_wrmse_s = torch.load(self.save_dir+'/wrmse_s_temp.pt', weights_only = True, map_location='cpu')
            
            torch.save(torch.cat((temp_real, x_next.detach().cpu()), dim=0), self.save_dir+'/realizations_temp.pt')
            if self.hard_data: torch.save(torch.cat((temp_wrmse_h, wrmse_h_t.detach().cpu()), dim=0), self.save_dir+'/wrmse_h_temp.pt')
            if self.seismic: torch.save(torch.cat((temp_wrmse_s, wrmse_s_t.detach().cpu()), dim=0), self.save_dir+'/wrmse_s_temp.pt')
            del temp_real
            
        except:
            torch.save(x_next.detach().cpu(), self.save_dir+'/realizations_temp.pt')
            if self.hard_data: torch.save(wrmse_h_t.detach().cpu(), self.save_dir+'/wrmse_h_temp.pt')
            if self.seismic: torch.save( wrmse_s_t.detach().cpu(), self.save_dir+'/wrmse_s_temp.pt')
            
        
        
        torch.cuda.empty_cache()
        time.sleep(1)
        del score_func, denoised,x_hat,latents
        if (self.seismic and not self.hard_data) : 
            return x_next, wrmse_s_t
        elif (not self.seismic and self.hard_data) : 
            return x_next, wrmse_h_t
        elif (self.seismic and self.hard_data): 
            return x_next, wrmse_s_t, wrmse_h_t
        else:
            return x_next
    
    
    def pltprgr(self,denoised,cond_grads,i):
        if i%1==0:
            n = 3 if (self.hard_data) & (self.seismic) else 2
            fig, axs = plt.subplots(n,3, figsize=(19,4*n))
            
            axs[0,1].set_title('predicted (mean)')
            x = axs[0,1].imshow(denoised.mean(0)[0].detach().cpu(), cmap='gray',
                                vmin=-1, vmax=1)
            plt.colorbar(x)
            x = axs[1,1].imshow(denoised.mean(0)[1].detach().cpu(), cmap='jet',
                                vmin=-1, vmax=1)
            plt.colorbar(x)

            if (self.hard_data):
                axs[0,0].set_title('conditioning')
                x = axs[0,0].imshow(2*(self.yh_obs*self.mask)[0,0].detach().cpu()-1, cmap='gray',
                                    vmin=-1, vmax=1)
                plt.colorbar(x)
                x= axs[1,0].imshow(2*(self.yh_obs*self.mask)[0,1].detach().cpu()-1,  cmap='jet',
                                    vmin=-1, vmax=1)
                plt.colorbar(x)
                if self.seismic:
                    axs[2,0].imshow(self.ys_obs[:,0].mean(0).detach().cpu(), cmap='seismic')
                    x_0hat_r_ip = ((denoised[:,1,None]+1)/2)*self.ip[2]+self.ip[0]
                    dhat = self.physics_forward(x_0hat_r_ip)
                    x = axs[2,1].imshow(dhat.mean(0).detach().cpu().squeeze(), cmap='seismic')
                    plt.colorbar(x)
            
            elif (not self.hard_data) & (self.seismic):
                axs[0,0].set_title('conditioning')
                x = axs[0,0].imshow(self.ys_obs[:,0].mean(0).detach().cpu(), cmap='seismic')
                plt.colorbar(x)
                x_0hat_r_ip = ((denoised[:,1,None]+1)/2)*self.ip[2]+self.ip[0]
                axs[1,0].set_title('predicted')
                dhat = self.physics_forward(x_0hat_r_ip)
                x = axs[1,0].imshow(dhat.mean(0).detach().cpu().squeeze(), cmap='seismic')
                plt.colorbar(x)
                
            axs[0,2].set_title('cond gradient (mean)')
            x = axs[0,2].imshow(cond_grads.mean(0)[0].detach().cpu(), cmap='hsv')
            plt.colorbar(x)
            x = axs[1,2].imshow(cond_grads.mean(0)[1].detach().cpu(), cmap='hsv')
            plt.colorbar(x)
            
            plt.show()
       
    
    def invert(self):
        realizations=torch.zeros(self.n_samples, self.image_depth, self.image_size[0],
                                 self.image_size[1] if self.image_size[1]!=100 else self.image_size[1]+4)
        
        wrmse_s = torch.zeros(self.n_samples,self.num_steps)
        wrmse_h = torch.zeros(self.n_samples,self.num_steps)
 
        N_max = 1 if self.seismic else 1
        if self.n_samples>N_max:
            for i in range((self.n_samples//N_max)):
                print(f'Realization {i+1}')
                if (self.seismic and not self.hard_data) : 
                    realizations_t, rmse_s_t = self.edm_sampler(N_max)
                    wrmse_s[i*N_max:i*N_max+N_max]=rmse_s_t
                    
                elif (not self.seismic and self.hard_data) : 
                    realizations_t, rmse_h_t = self.edm_sampler(N_max)
                    wrmse_h[i*N_max:i*N_max+N_max] = rmse_h_t
                
                elif (self.seismic and self.hard_data): 
                    realizations_t, rmse_s_t, rmse_h_t = self.edm_sampler(N_max)
                    wrmse_s[i*N_max:i*N_max+N_max]=rmse_s_t
                    wrmse_h[i*N_max:i*N_max+N_max]=rmse_h_t
                    
                realizations[i*N_max:i*N_max+N_max]=realizations_t.detach().cpu()
            
        else:
            if (self.seismic and not self.hard_data) : realizations, wrmse_s = self.edm_sampler(self.n_samples)
            elif (not self.seismic and self.hard_data) : realizations, wrmse_h = self.edm_sampler(self.n_samples)
            elif (self.seismic and self.hard_data): realizations, rmse_s, wrmse_h = self.edm_sampler(self.n_samples)
        
        realizations = (realizations+1)/2
        realizations[:,1] = (realizations[:,1])*(self.ip[1].item()-self.ip[0].item())+self.ip[0].item()
        realizations = realizations.detach().cpu().squeeze()

        self.realizations = realizations
        self.wrmse_s= wrmse_s
        self.wrmse_h= wrmse_h
        
        torch.save(self.realizations, self.save_dir+'/reals.pt')
        torch.save(self.wrmse_s, self.save_dir+'/wrmse_s.pt')
        torch.save(self.wrmse_h, self.save_dir+'/wrmse_h.pt')

parser = argparse.ArgumentParser()
parser.add_argument("--workdir", type=str, default= '//')
parser.add_argument("--save_dir", type=str, default= '//')
parser.add_argument("--train_data_dir", type=str, default= '/')
parser.add_argument("--test_models_dir", type=str, default= '/')
parser.add_argument("--net_dir", type=str, default= '/')
parser.add_argument("--net_snapfile", type=str, default= '/network-snapshot-001800.pkl') 

parser.add_argument("--test_model_n", type=str, default='1')
parser.add_argument("--test_model_folder", type=str, default= 'Test_models_DGMs/')
parser.add_argument("--wavelet_file", type=str, default= 'wavelet.asc')
parser.add_argument("--image_size", type= int, default=[80,100])
parser.add_argument("--image_depth", type= int, default=2)
parser.add_argument("--sigma_max", type= int, default=80)
parser.add_argument("--rho", type= float, default=7,    
                    help='determines the noise schedule, 1 is linear. 7 is the optimal exponential trend from Karras EDM paper')  
parser.add_argument("--device", type= str, default="cuda:3")       

parser.add_argument("--num_steps", type= int, default= n_steps)
parser.add_argument("--n_samples", type= int, default= 100)

parser.add_argument("--hard_data_error", default= [.1*noise_factor,.05*noise_factor], help='Sigma for [Facies, Ip], normalized between 0 and 1')
parser.add_argument("--hd_cond_where", default= [30, 50])
parser.add_argument("--relative_data_error", default= .05*noise_factor)
parser.add_argument("--absolute_data_error", default= 1*noise_factor)
parser.add_argument("--error_x0", default= error, 
help= 'dps / prop / actual_dps: 1) the way it is theoretically proposed / our correction / how it is implemented in their DDPM implementation')

parser.add_argument("--seismic", type=bool, default=True, help='True if conditioning on sesimic')
parser.add_argument("--hard_data", type=bool, default=False, help='True if conditioning on well data')

args = parser.parse_args()

print(args)

sys.path.append(args.net_dir+'/Code_backup') #torch.utils is necessary to load the model
import os
from dataset import FaciesSet

# using training data to compute the error
if args.error_x0=='prop':
    dataset = FaciesSet(args.train_data_dir, args.image_size, args.image_depth)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    for i, data in enumerate(loader):
        test_data = data[0]
        del data
        break


os.makedirs(args.save_dir, exist_ok=True)

DPS = DPS_Inversion(args)
if args.error_x0=='prop': 
    print('Error estimation per timestep')
    DPS.compute_estimation_error(test_data.to(args.device))

DPS.invert()
            
try:
    exists = DPS.realizations
except:
    DPS.realizations = torch.load(args.save_dir+'/reals.pt', weights_only=True)
    DPS.wrmse_s = torch.load(args.save_dir+'/wrmse_s.pt', weights_only=True)
    DPS.wrmse_h = torch.load(args.save_dir+'/wrmse_h.pt', weights_only=True)

DPS.results_analysis()
            
