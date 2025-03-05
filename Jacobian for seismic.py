# -*- coding: utf-8 -*-
"""
Spyder Editor
This piece of code is to compute the Jacobian for the 1D convolution case, in fullstack seismic
AI = acoustic impedance : 1D tensor of size n
RC = reflectivity coefficient : 1D tensor of size n-1

RC = (AI_i+1 - AI_i)/(AI_i+1 + AI_i) 

#the jacobian is the convolution of the known wavelet with the Jacobian of RC

This is a temporary script file.
"""
import torch
import matplotlib.pyplot as plt

frequency = 1/(torch.pi*5)
t= torch.arange(-21,20,1)
omega = torch.pi * frequency
wavelet = (1 - 2 * (omega * t) ** 2) * torch.exp(-(omega * t) ** 2)
padding = (len(wavelet)//2,0)

convolution = torch.nn.Conv2d(1,1, kernel_size=1, padding= padding, bias=False)
convolution.weight = torch.nn.Parameter(wavelet[None,None,:,None]).requires_grad_(False)

# %% Jacobian of RC
def diag_els (xi, xii):
    den = (xi+xii)**2
    num1 = -2*xii
    num2 = 2*xi
    
    return torch.tensor([num1/den, num2/den])

AI = torch.tensor([1,2,3,4,5]).float()
n = len(AI)


JRC_struct = torch.zeros(n,n)
for i in range(n-1):
    JRC_struct[i,i:i+2]= diag_els(AI[i], AI[i+1])
JRC_struct[i+1,i+1]= diag_els(AI[i], AI[i])[0] #last element is repeated

plt.imshow(JRC_struct)


# %% Convolution 
frequency = 1/(torch.pi*5)
t= torch.arange(-21,20,1)
omega = torch.pi * frequency
wavelet = (1 - 2 * (omega * t) ** 2) * torch.exp(-(omega * t) ** 2)
padding = (len(wavelet)//2,0)
convolution = torch.nn.Conv2d(1,1, kernel_size=1, padding= padding, bias=False)
convolution.weight = torch.nn.Parameter(wavelet[None,None,:,None]).requires_grad_(False)


AI = torch.tensor([1,1,1,1,1,3,3,3,3,3]).float()
RC = (AI[1:]-AI[:-1]) / (AI[1:]+AI[:-1]) 
RC = torch.cat((RC,torch.zeros(1)))
seismic1 = convolution(RC[None,None,:,None])
print(AI.shape,RC.shape,seismic1.shape)

plt.plot(seismic1.squeeze())
plt.plot(AI.squeeze())
plt.plot(RC.squeeze())
plt.show()


AI = AI[None,None,:,None]
AI = torch.cat((AI, AI[:,:,[-1],:]), dim=2) # repeats last element
ip_d =  AI[:, :, 1:, :] - AI[:, :, :-1, :]
ip_a = (AI[:, :, 1:, :] + AI[:, :, :-1, :])    
rc = ip_d / ip_a
seismic2 = convolution(rc)
print(AI.shape,rc.shape,seismic2.shape)
plt.plot(seismic2.squeeze())
plt.plot(AI.squeeze())
plt.plot(rc.squeeze())
plt.show()


AI = torch.tensor([1,1,1,1,1,3,3,3,3,3]).float()
RC = (AI[1:]-AI[:-1]) / (AI[1:]+AI[:-1]) 
RC = torch.cat((RC,torch.zeros(1)))

padded_n = RC.shape[0]+len(wavelet)
padded_rc = torch.cat((RC, torch.zeros(len(wavelet)//2 + len(wavelet)%2)))
padded_rc = torch.cat((torch.zeros(len(wavelet)//2),padded_rc))

toeplitz_w = torch.zeros(padded_n-len(wavelet)+1-len(wavelet)%2, padded_n) #padding included
for i in range(toeplitz_w.shape[0]):
    toeplitz_w[i,i:i+len(wavelet)]=wavelet

seismic3 = toeplitz_w @ padded_rc[None,].T
print(AI.shape,RC.shape,seismic3.shape)

plt.plot(seismic3.squeeze())
plt.plot(AI.squeeze())
plt.plot(RC.squeeze())
plt.show()

# %%

def ref_coeff(AI, wavelet_lenght):
    #computes the reflectivity coefficients, increases size by 1
    RC = (AI[1:]-AI[:-1]) / (AI[1:]+AI[:-1]) 
    RC = torch.cat((RC,torch.zeros(1)))
    padded_rc1 = torch.cat((RC, torch.zeros(wavelet_lenght//2 + wavelet_lenght%2)))
    padded_rc = torch.cat((torch.zeros(wavelet_lenght//2),padded_rc1))
    return padded_rc


def diag_els (xi, xii):
    #Jacobian of reflectivity coefficients  models
    den = (xi+xii)**2
    num1 = -2*xii
    num2 = 2*xi
    return torch.tensor([num1/den, num2/den])

def conv_matrix(trace_lenght, wavelet):
    #creates the matrix for 1D convolution, input data is padded
    wavelet_lenght = len(wavelet)
    trace_lenght = trace_lenght+wavelet_lenght
    toeplitz_w = torch.zeros(trace_lenght-wavelet_lenght+1-wavelet_lenght%2, 
                             trace_lenght) 
    
    for i in range(toeplitz_w.shape[0]):
        toeplitz_w[i,i:i+len(wavelet)]=wavelet
        
    return toeplitz_w


#create / load wavelet
frequency = 1/(torch.pi*5)
t= torch.arange(-20,20,1)
omega = torch.pi * frequency
wavelet = (1 - 2 * (omega * t) ** 2) * torch.exp(-(omega * t) ** 2)
padding = (len(wavelet)//2,0)
convolution2 = torch.nn.Conv2d(1,1, kernel_size=1, padding= padding, bias=False)
convolution2.weight = torch.nn.Parameter(wavelet[None,None,:,None]).requires_grad_(False)

#input data
AI = torch.tensor([1,1,1,1,1,1,3,3,3,3]).float()
n = len(AI)

#seismic model
convolution = conv_matrix(n, wavelet)

#get reflectivity
RC = ref_coeff(AI, len(wavelet))

#get seismic
seis1 = convolution @ RC


#get Jacobian of RC
JRC_struct = torch.zeros(len(RC),len(RC))
for i in range(n-1):
    idx = i+len(wavelet)//2
    JRC_struct[idx,idx:idx+2]= diag_els(AI[i], AI[i+1])
idx = i+1+len(wavelet)//2
JRC_struct[idx,idx]= diag_els(AI[i], AI[i])[0] #last element is repeated

J1 = (convolution @ JRC_struct)[:-1, len(wavelet)//2:-len(wavelet)//2]


def physics_forward(realization):
    
    ip = torch.cat((realization, realization[:,:,[-1],:]), dim=2) # repeats last element
    ip_d =  ip[:, :, 1:, :] - ip[:, :, :-1, :]
    ip_a = (ip[:, :, 1:, :] + ip[:, :, :-1, :])    
    rc = ip_d / ip_a
    
    return convolution2(rc).squeeze()

seis2 = physics_forward(AI[None,None,:,None])

J2 = torch.autograd.functional.jacobian(physics_forward, AI[None,None,:,None]).reshape(11,10)

AI.requires_grad_(True)

def convo(inp):
    RC1 = (inp[1:]-inp[:-1]) / (inp[1:]+inp[:-1]) 
    RC = torch.cat((RC1,torch.zeros(1)))
    padded_rc1 = torch.cat((RC, torch.zeros(40//2 + 40%2)))
    padded_rc = torch.cat((torch.zeros(40//2),padded_rc1))
    return (convolution @ padded_rc)[:AI.shape[0]]
    
J3 = torch.autograd.functional.jacobian(convo, AI).reshape(10,10)

plt.imshow(J1)
plt.imshow(J2)
plt.imshow(J3)

plt.plot(seis1)
plt.plot(seis2)
