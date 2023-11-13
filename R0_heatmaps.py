import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import math

plt.rcParams['font.size'] = '18'
plt.rcParams.update({'axes.labelsize': 'large'})

'Define the parameters as described in Sections 2.2 and 2.3'
phiT = 2
phiH = 1
varphi = 1/6
range_a=[10**-6,2*10**-5,10**-4] # values of alpha
nu = 10**-2
mu = 10**-3


'Define the ranges of gamma1 (x-axis) and beta1 (y-axis)'    
X0 = 0
X1 = 10**-2
Y0 = 0
Y1 = 10**-5

N = 50 # size of the grid

gamma1s = np.linspace(X0,X1,N+1)
beta1s = np.linspace(Y0,Y1,N+1)
X,Y = np.meshgrid(gamma1s,beta1s)

R0s_list =[] # list to store the R0s array for each value of alpha

# lists to store the minimum and maxiumum value of R0 for each heatmap, needed to define a sensible colormap range
maximum_R0s = []
minimum_R0s = []

'Calculate the R0 values to be plotted in the heatmaps'
for alpha in range_a:
    RTT = alpha * phiT/nu/nu
    print('\u03B1 =',alpha, 'R_TT =', RTT)
    
    R0s = np.zeros((N+1,N+1))
    
    for i, gamma1 in enumerate(gamma1s):
        for j, beta1 in enumerate(beta1s):            
            RHT = gamma1 * phiT/(nu*(mu+varphi))
            RTH = beta1 * phiH/nu/mu
            R0s[j,i] = 1.0 / 2.0 * (RTT + math.sqrt(RTT**2 + 4 * RTH * RHT))
    
    R0s_list.append(R0s)
    maximum_R0s.append(np.max(R0s))
    minimum_R0s.append(np.min(R0s))
    
'Define the maximum and the minimum of the colormap'
my_cmap_max = max(maximum_R0s)
my_cmap_min = min(minimum_R0s)
ColorSpectrum = np.logspace(np.log10(my_cmap_min),np.log10(my_cmap_max),1000)
my_cmap = 'magma'
    
'Plot the heatmaps'
fig, axes = plt.subplots(1, 3, figsize=(10,3.5), sharex=True, sharey=True, dpi=300)
plt.subplots_adjust(wspace=0.32)

for k, alpha in enumerate(range_a, 1):
    plt.subplot(1,3,k)
    plt.xlabel(r'$\gamma_1$')
    if k ==1:
        plt.ylabel(r'$\beta_1$')
    HM = plt.contourf(X, Y, R0s_list[k-1], levels=ColorSpectrum, cmap=my_cmap, norm = LogNorm(), extend='both')
    
    plt.contour(X,Y,R0s_list[k-1],levels=[1],colors='black') # black line where R0 is 1

    plt.xticks([0,10**-2*0.5,10**-2*1])
    plt.yticks([0,10**-5*0.5,10**-5*1])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

fig.subplots_adjust(right=0.95)
cbar_ax = fig.add_axes([0.97, 0.15, 0.04, 0.7])
fig.colorbar(HM, label='$R_0$', cax=cbar_ax, shrink=0.5, ticks=[0.1, 1, 5])
plt.savefig('R0_single_infection.png', bbox_inches = 'tight')

