# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:09:32 2025

@author: rmiele
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:09:32 2025

@author: rmiele
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import torch.nn.functional as F
from tqdm import tqdm
import time


#wavelet as defined in DPS inversion
frequency = 1/(torch.pi*5)
t= torch.arange(-20,20,1)
omega = torch.pi * frequency
wavelet = ((1 - 2 * (omega * t) ** 2) * torch.exp(-(omega * t) ** 2))*100

wavelet = np.expand_dims(wavelet, 0) # add bacth [B x H x W x C]
if wavelet.ndim==2: 
    wavelet= np.expand_dims(wavelet, 0)
    wavelet= np.expand_dims(wavelet, -1)
wavelet = torch.from_numpy(wavelet).double().to('cpu')
k = wavelet.shape[-2]
padding = (k//2,0)
seismic_conv = torch.nn.Conv2d(1,1, kernel_size=1, padding= padding, bias=False)
seismic_conv.weight = torch.nn.Parameter(wavelet).requires_grad_(False)

def physics_forward (ip):
    ip = torch.cat((ip, ip[:,:,[-1],:]), dim=2) # repeats last element
    ip_d =  ip[:, :, 1:, :] - ip[:, :, :-1, :]
    ip_a = (ip[:, :, 1:, :] + ip[:, :, :-1, :])    
    rc = ip_d / ip_a
    return seismic_conv(rc)[:,:,:80,:]

def add_cbar(i,j=False, ax=False, fig= None, axs=None):
    if j: bbox= axs[i,j].get_position()
    else: bbox= axs[i].get_position()
    cax = fig.add_axes((bbox.x1+0.01, bbox.y0, 0.02, bbox.y1-bbox.y0))        
    fig.colorbar(ax, cax=cax, orientation='vertical')


test_models_folder = 'C:/Users/rmiele/OneDrive - Université de Lausanne/Codes/Inversion/Test_models/'
results_folder = 'C:/Users/rmiele/OneDrive - Université de Lausanne/Codes/Inversion/DPS_EDM/Save/Seismic_dps/'
save_dir = 'C:/Users/rmiele/OneDrive - Université de Lausanne/Codes/Inversion/DPS_EDM/Save/Data_analysis/Seismic_prop'
sigma_ys_asb = 1
sigma_ys_rel = .05

os.makedirs(save_dir, exist_ok=True)

#log_rmse
rmse_s = torch.load(results_folder+'/rmse_s.pt', weights_only=True)
rmse_h = torch.load(results_folder+'/rmse_h.pt', weights_only=True)

#real data
true_Ip= torch.load(test_models_folder+f'/mtest_ip1.pt', weights_only=True, map_location='cpu')
true_F= torch.load(test_models_folder+f'/mtest_facies1.pt', weights_only=True, map_location='cpu')
yh_obs = torch.cat((true_Ip[None,], true_F[None,]), dim=0)

ys_obs = physics_forward(true_Ip[None,None,:].double())


sigma_ys = sigma_ys_rel*torch.abs(ys_obs.detach().clone())+sigma_ys_asb


posterior = torch.load(results_folder+'reals.pt', weights_only=True)
posterior_F = posterior[:,0,:]
posterior_Ip = posterior[:,1,:]
posterior_S = physics_forward(posterior_Ip[:,None,:].double())
del posterior, true_Ip, true_F


#plot rmse ------------------------------------------------------------------------------
plt.figure()
plt.plot(rmse_s.T, color='k')
plt.plot(rmse_s.T[:,0],label='Samples', color='k')
plt.plot(np.arange(0,rmse_s.shape[1]), 
         np.zeros(rmse_s.shape[1])+sigma_ys.mean().item(),
         linestyle='--', color='r', label=r'$\sigma_ys$')
plt.ylim([0,rmse_s.max()+0.1])
plt.ylabel(r'RMSE')
plt.xlabel('Steps')
plt.legend()
plt.savefig(save_dir+'/rmse_s.png',bbox_inches='tight')
plt.show()

#plot global seismic data error ------------------------------------------------------------------------------

fig, axs= plt.subplots(1,3, figsize=(5.5,4))
ax= axs[0].imshow(ys_obs.squeeze(), cmap='seismic')
axs[0].set_title('True')

e_hat = (ys_obs - posterior_S).mean(0).squeeze()
e = (ys_obs - posterior_S).squeeze()
ax = axs[1].imshow(e_hat,cmap='seismic')
axs[1].set_title('Avg Residuals\n30 samples')
add_cbar(i=1,ax=ax,fig=fig, axs=axs)
axs[1].set_yticklabels([])
axs[2].hist(e.flatten(), density=True, 
            weights= np.ones(len(e.flatten()))/len(e.flatten()),
            bins=30)
axs[2].set_xlim([-5,5])
axs[2].set_title('Error distribution') # \n Std. dev. {(e.flatten().std()/8000).round(decimals=3).item()}')
bbox= axs[2].get_position()
bbox.x0+=0.2; bbox.x1+=0.2; bbox.y0+=0.2; bbox.y1-=0.2
axs[2].set_position(bbox)
plt.savefig(save_dir+f'/Seismic_summary.png',bbox_inches='tight')
plt.show()

#trace seismic data error ------------------------------------------------------------------------------
plt.figure()
plt.plot(e[:,:,50].T, c='gray', alpha=0.5)
plt.plot(e[0,:,50].T, c='gray', alpha=0.5, label=r'$\bf{d_{obs}}-\bf{\hat{d}}$')
plt.plot(sigma_ys.squeeze()[:,50], c='r')
plt.plot(-sigma_ys.squeeze()[:,50], c='r', label= r'$\sigma\bf{_d}$')
plt.legend()
plt.savefig(save_dir+f'/Seismic_error_trace.png',bbox_inches='tight')

plt.show()


plt.figure()
plt.plot(ys_obs[0,0,:,50].T, c='k', label=r'$\bf{d_{obs}}$', zorder=2)
plt.plot(posterior_S[:,0,:,50].T, c='gray', alpha=0.7, zorder=1)
plt.plot(posterior_S[0,0,:,50].T, c='gray', alpha=0.7, label=r'$\bf{\hat{d}}$', zorder=1)
plt.plot(ys_obs[0,0,:,50].T+sigma_ys.squeeze()[:,50], c='r', zorder=1)
plt.plot(ys_obs[0,0,:,50].T-sigma_ys.squeeze()[:,50], c='r', label= r'$\bf{d_{obs}}±\sigma\bf{_d}$', zorder=1)
plt.legend()
plt.savefig(save_dir+f'/Seismic_trace.png',bbox_inches='tight')

plt.show()



"""
        ys_obs = (self.ys_obs- self.noise_ys)[:,:,:,:100].detach().cpu().squeeze()
        yh_obs = (self.yh_obs- self.noise_yh)[:,:,:,:100].detach().cpu().squeeze()
        
        yh_obs[1] = yh_obs[1]*(self.ip[1].item()-self.ip[0].item())+self.ip[0].item()
        sigma_ys = self.sigma_ys.detach().cpu().squeeze()
        sigma_yh = self.sigma_yh.detach().cpu().squeeze()
        
        N = self.image_size[0]*self.image_size[1]
        
        if (self.seismic and not self.hard_data):             text='seismic only'
        if (not self.seismic and self.hard_data):             text='hd only'
        if (self.seismic and self.hard_data):                 text='both'
        
        #RESULTS ANALYSIS
            

        
        if self.hard_data :
            #plot rmse -----------------------------------------------------------------------------------------
            plt.figure()
            plt.plot(self.rmse_h_t.T, color='k')
            plt.plot(self.rmse_h_t.T[:,0],label='Samples', color='k')
            plt.plot(np.arange(0,self.num_steps), 
                     np.zeros(self.num_steps)+sigma_yh[1].mean().item(),
                     linestyle='--', color='r', label=r'$\sigma_ys$')
            plt.ylim([0,self.rmse_h_t.max()+0.1])
            plt.ylabel(r'RMSE')
            plt.xlabel('Steps')
            plt.legend()
            plt.savefig(self.save_dir+f'//rmse_h_{text}.png',bbox_inches='tight')
            plt.show()

        for i in range(self.image_depth):
            
            typ = 'Fac' if i==0 else 'Ip'
            plt.figure()
            if not self.hard_data:
                plt.title('Not conditioning - Blind well')
            plt.plot(yh_obs[i,:,50],
                     np.linspace(80,0,80), c='r', zorder=2, label='Conditioning')
            plt.plot(self.realizations[:,i,:,50].T, 
                     np.tile(np.linspace(80,0,80), (self.realizations.shape[0],1)).T, c='gray', zorder=1)
            plt.plot(self.realizations[0,i,:,50], 
                     np.linspace(80,0,80), c='gray', zorder=1, label='realizations')
            plt.savefig(self.save_dir+f'//{typ}_conditioning_Well_{text}.png',bbox_inches='tight')
            plt.show()
            
        # Plot facies and ip ------------------------------------------------------------------------------------
        import matplotlib
        yh_obs_0 = np.ma.array (yh_obs[0], mask=yh_obs[0]<-1.5)
        yh_obs_1 = np.ma.array (yh_obs[1], mask=yh_obs[1]<-1.5)
        cmap0 = matplotlib.cm.gray
        cmap0.set_bad('gray',1.)

        cmap1 = matplotlib.cm.jet
        cmap1.set_bad('gray',1.)

        n_raws = 3 if (self.hard_data) and (self.seismic) else 2
    
        fig, axs= plt.subplots(n_raws, 4, figsize = (18,n_raws*4), sharex=True, sharey=True)
        axs[0,0].set_title('True')
        axs[0,0].set_ylabel('Facies')
        ax = axs[0,0].imshow(self.true_F.detach().cpu().squeeze(),cmap='gray')
        axs[1,0].imshow(self.true_Ip.detach().cpu().squeeze(),cmap='jet',vmin = self.ip[0], vmax = self.ip[1])
        axs[1,0].set_ylabel('Ip')
        axs[0,1].set_title('Observed')
        
        axs[0,2].set_title(f'Inferred (mean)\n{self.n_samples} samples')
        ax = axs[0,2].imshow(self.realizations[:,0].mean(0),cmap='gray',vmin=0,vmax=1, interpolation='none')
        add_cbar(0,2,ax,fig=fig, axs=axs)
        ax1 = axs[1,2].imshow(self.realizations[:,1].mean(0),cmap='jet', vmin = self.ip[0], vmax = self.ip[1], interpolation='none')
        
        axs[0,3].set_title(f'Inferred (Std. dev.)\n{self.n_samples} samples')
        ax = axs[0,3].imshow(self.realizations[:,0].std(0),cmap='gray', interpolation='none')
        bbox= axs[0,3].get_position()
        bbox.x0+=0.07; bbox.x1+=0.07
        axs[0,3].set_position(bbox)
        add_cbar(0,3,ax,fig=fig, axs=axs)
        
        ax = axs[1,3].imshow(self.realizations[:,1].std(0),cmap='jet', interpolation='none')
        bbox= axs[1,3].get_position()
        bbox.x0+=0.07; bbox.x1+=0.07
        axs[1,3].set_position(bbox)
        add_cbar(1,3,ax,fig=fig, axs=axs)

        if self.hard_data:
            axs[0,1].imshow(yh_obs_0, cmap=cmap0,vmin=0,vmax=1, interpolation='none')#, vmin = 0, vmax =1)
            axs[1,1].imshow(yh_obs_1,cmap=cmap1, vmin = self.ip[0], vmax = self.ip[1],interpolation='none')

            if self.seismic:
                axs[2,0].set_visible(False)
                axs[2,1].set_ylabel('Seismic data')
                axs[2,1].imshow(ys_obs,cmap='seismic',vmin=self.yslims[0],vmax=self.yslims[1])
                ax = axs[2,2].imshow(dhat.mean(0),cmap='seismic',vmin=self.yslims[0],vmax=self.yslims[1])
                add_cbar(2,2,ax,fig=fig, axs=axs)
                
                ax = axs[2,3].imshow(dhat.std(0),cmap='seismic',vmin=self.yslims[0],vmax=self.yslims[1])
                
                bbox= axs[2,3].get_position()
                bbox.x0+=0.07; bbox.x1+=0.07
                axs[2,3].set_position(bbox)
                add_cbar(2,3,ax,fig=fig, axs=axs)
        
        elif self.seismic:
            
            axs[0,1].imshow(ys_obs, cmap='seismic', vmin=self.yslims[0], vmax=self.yslims[1])
            
            axs[1,1].set_title('Inferred data error (mean)')
            ax = axs[1,1].imshow(e_hat, cmap='seismic', vmin=self.yslims[0], vmax=self.yslims[1])
            bbox= axs[1,1].get_position()
            bbox.x0-=0.02; bbox.x1-=0.02
            axs[1,1].set_position(bbox)
            add_cbar(1,1,ax,fig=fig, axs=axs)
            
            bbox= axs[0,1].get_position()
            bbox.x0-=0.02; bbox.x1-=0.02
            axs[0,1].set_position(bbox)
            
            bbox= axs[1,2].get_position()
            bbox.x0+=0.01; bbox.x1+=0.01
            axs[1,2].set_position(bbox)
            
            bbox= axs[0,2].get_position()
            bbox.x0+=0.01; bbox.x1+=0.01
            axs[0,2].set_position(bbox)
            
            ax = axs[1,2].imshow(self.realizations[:,1].mean(0),cmap='jet', 
                                 vmin = self.ip[0], vmax = self.ip[1], 
                                 interpolation='none')
        add_cbar(1, 2, ax1, fig=fig, axs=axs)
        
        plt.savefig(self.save_dir+f'//Fac_Ip_summary_{text}.png',
                    bbox_inches='tight')
        plt.show()
        
     
        #plot Ip distribution comparison ------------------------------------------------------------------------------
        plt.figure()
        plt.hist(self.true_Ip.flatten().detach().cpu(), 
                 bins=np.linspace(self.ip[0].item(),self.ip[1].item(),30),
                 density=True, weights= np.ones(N)/N, 
                 alpha=.7)

        plt.hist(self.realizations[:,1].flatten().detach().cpu(), 
                 bins=np.linspace(self.ip[0].item(),self.ip[1].item(),30),
                 density=True, weights= np.ones(self.n_samples*N)/(self.n_samples*N),
                 alpha=.7)
        plt.savefig(self.save_dir+f'//Ip_summary_{text}.png',bbox_inches='tight')
        plt.show()        
        torch.save(self.realizations, self.save_dir+'/reals.pt')
        

"""