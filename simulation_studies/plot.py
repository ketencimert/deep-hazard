# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:30:35 2022

@author: Mert
"""
import torch
import numpy as np
import matplotlib.pyplot as plt



def weibull_hazard(x, t):
    lambda_ = 1/x**2
    scale = 1/lambda_
    k = 1.5 * torch.sigmoid(x+10)
    return (scale**k * k * t ** (k-1)).view(n,n).detach().cpu().numpy()

def weibull_survival(x, t):
    lambda_ = 1/x**2
    scale = 1/lambda_
    k = 1.5 * torch.sigmoid(x+10)
    return torch.exp(- scale * t ** k).view(n,n).detach().cpu().numpy()

def weibull_pdf(x, t):
    lambda_ = 1/x**2
    scale = 1/lambda_
    k = 1.5 * torch.sigmoid(x+10)
    return (k/lambda_ * (t / lambda_) ** (k-1) * torch.exp(-(t / lambda_)**k)
            ).view(n,n).detach().cpu().numpy()

n = 500

# import matplotlib.pyplot as plt

# best_lambdann.eval()
# n = 100
# k = 0
# x1 = np.linspace(-3, 3, n)
# x2 = np.linspace(-3, 3, n)
# x = torch.cat([torch.tensor(x).view(-1,1) for x in np.meshgrid(x1, x2)], -1).to(args.device)
# t = torch.ones_like(x)[:,0]

# real_hazard = torch.exp(-(x[:,0]**2 + x[:,1]**2)).view(n,n).detach().cpu().numpy()
# real_hazard = real_hazard + 10
# hazard = best_lambdann(x=x,t=k*t).view(n,n).detach().cpu().numpy()

# xx, yy = np.meshgrid(x1, x2)
# h = plt.contourf(xx, yy, real_hazard)
# h = plt.contourf(xx, yy, hazard)

# plt.axis('scaled')
# plt.colorbar()
# plt.show()

fig, axs = plt.subplots(2, 2)

x = np.linspace(-2, 2, n)
t = np.linspace(0, 10, n)
xt = torch.cat([torch.tensor(x).view(-1,1) for x in np.meshgrid(x, t)], -1)
pdf = weibull_pdf(xt[:,0], xt[:,1])
surv = weibull_survival(xt[:,0], xt[:,1])
haz = weibull_hazard(xt[:,0], xt[:,1])
xx, tt = np.meshgrid(x, t)
axs[0,0].contourf(xx, tt, pdf, cmap='jet')
axs[0,0].set_title('PDF')
axs[0,0].set_xlabel('covariate x')
axs[0,0].set_ylabel('time t')
axs[0,1].contourf(xx, tt, surv, cmap='jet')
axs[0,1].set_title('Survival')
axs[0,1].set_xlabel('covariate x')
axs[0,1].set_ylabel('time t')
axs[1,0].contourf(xx, tt, haz, cmap='jet')
axs[1,0].set_title('Hazard')
axs[1,0].set_xlabel('covariate x')
axs[1,0].set_ylabel('time t')
plt.tight_layout()

# x = torch.linspace(-2, 1, n)
# lambda_ = 1/torch.sin(x+2.06)**2
# scale = 1/lambda_
# k = 1.5 * torch.sigmoid(x+10)

# pdfs = []
# for t in torch.linspace(1, 10, n):
#     pdfs.append(
#         Weibull(scale=1/scale, concentration=k).log_prob(t).view(-1).exp().view(-1,1)
#         )

# pdfs = torch.cat(pdfs, -1).numpy()
# x = np.linspace(-2, 1, n)
# t = np.linspace(1, 10, n)
# tt, xx = np.meshgrid(t, x)
# h = plt.contourf(xx, tt, pdfs)

# best_lambdann.eval()
# n = 300
# x = torch.linspace(-2, 2, n).to(args.device).view(-1,1)
# hazards = []
# for t in torch.linspace(0, 10, n).to(args.device):
#     hazards.append(
#         best_lambdann(x=x,t=torch.ones_like(x)*t).view(-1,1).detach().cpu()
#         )
# hazards = torch.cat(hazards, -1).numpy()
# x = np.linspace(-2, 2, n)
# t = np.linspace(1, 10, n)
# tt, xx = np.meshgrid(t, x)
# h = plt.contourf(xx, tt, hazards)



# n = 100
# k = 0
# x1 = np.linspace(-5, 5, n)
# x2 = np.linspace(-5, 5, n)
# x = torch.cat([torch.tensor(x).view(-1,1) for x in np.meshgrid(x1, x2)], -1).to(args.device).float()
# t = 0
# hazard = best_lambdann(
#     x=x,
#     t=torch.ones_like(x[:,0])*t).view(-1,1).view(n,n).detach().cpu()
# x1 = np.linspace(-3, 3, n)
# x2 = np.linspace(-3, 3, n)
# xx, yy = np.meshgrid(x1, x2)
# h = plt.contourf(xx, yy, hazard)



# h = plt.contourf(xx, yy, hazard)

# # x = np.linspace(-5, 5, 10)
# # y = np.linspace(-5, 5, 10)
# # # full coordinate arrays
# # xx, yy = np.meshgrid(x, y)
# # zz = np.sqrt(xx**2 + yy**2)
# # xx.shape, yy.shape, zz.shape
# # ((101, 101), (101, 101), (101, 101))
# # sparse coordinate arrays
