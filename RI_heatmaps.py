import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import eig
from matplotlib.colors import LogNorm

plt.rcParams['font.size'] = '18'
plt.rcParams.update({'axes.labelsize': 'large'})

# Note that the parameters epsilon1 and epsilon2 are denoted here by e1 and e2, respectively.
my_cmap = 'magma'


'Function to compute the invasion reproduction number derived in Eq. (10)'

def RI(fixed_params, e, alpha2):
    Phi, nu, alpha1 = fixed_params
    nu0 = nu1 = nu2 = nuc = nu
    ec = e
    delta1 = alpha1/2
    delta2 = alpha2/2
    alphac = delta1 + delta2
    p1 = delta1/alphac
    
    # Defining the strain 1 endemic equilibrium
    m0bar = nu/alpha1
    m1bar = Phi/nu-m0bar
    
    # Defining the elements of the next generation matrix (see Section 3.2)
    R22 = alpha2*m0bar/(alpha1*m1bar + nu2) + alpha1*m1bar*alphac*(1-p1)*(1-ec)*m0bar/(nuc*(alpha1*m1bar + nu2))
    Rc2 = alphac*(1-p1)*(1-ec)*m0bar/nuc
    R2c = alpha2*m1bar/(alpha1*m1bar + nu2) + alpha1*m1bar*(alphac*ec*m0bar + alphac*(1-p1*(1-ec))*m1bar)/(nuc*(alpha1*m1bar + nu2))
    Rcc = (alphac*ec*m0bar + alphac*(1-p1*(1-ec))*m1bar)/nuc
    
    return 1/2*(R22+Rcc + np.sqrt((R22+Rcc)**2-4*(R22*Rcc - Rc2*R2c)))


'Function to compute the normalised invasion reproduction number introduced in Eq. (11)'

def normed_RI(fixed_params, e, alpha2):
    Phi, nu, alpha1 = fixed_params
    
    # What the RI is under the limit as V2 tends to V1
    RI_neutral = RI(fixed_params, e, alpha1)
    
    return RI(fixed_params, e, alpha2)/RI_neutral


'Functions to compute the invasion reproduction number in the within-host model defined in Eq. (12)'

def phi_ij(alphai, alphaj):
    if alphai<=alphaj:
        return 0
    else:
        return 1
        
def within_host_RI(fixed_params, e, alpha2):
    Phi, nu, alpha1 = fixed_params
    nu0 = nu1 = nu2 = nuc = nu
    ec = e
    delta1 = alpha1/2
    delta2 = alpha2/2
    alphac = delta1 + delta2
    p1 = delta1/alphac
    
    phi12 = phi_ij(alpha1, alpha2)
    phi21 = phi_ij(alpha2, alpha1)
    
    # Defining the strain 1 endemic equilibrium
    m0bar = nu/alpha1
    m1bar = Phi/nu-m0bar
    
    # Defining the elements of the next generation matrix (see Appendix B)
    b11 = alpha2*m0bar/(alpha1*phi12*m1bar + nu2) + alpha1*phi12*m1bar*alphac*(1-p1)*(1-ec)*m0bar/(nuc*(alpha1*phi12*m1bar + nu2))
    b12 = alphac*(1-p1)*(1-ec)*m0bar/nuc
    b21 = alpha2*phi21*m1bar/(alpha1*phi12*m1bar + nu2) + alpha1*phi12*m1bar*(alphac*ec*m0bar + alphac*(1-p1*(1-ec))*phi21*m1bar)/(nuc*(alpha1*phi12*m1bar + nu2))
    b22 = (alphac*ec*m0bar + alphac*(1-p1*(1-ec))*phi21*m1bar)/nuc
    b11 = round(b11, 4)
    b12 = round(b12, 4)
    b21 = round(b21, 4)
    b22 = round(b22, 4)
    return 1/2*(b11+b22 + np.sqrt((b11+b22)**2-4*(b11*b22 -b12*b21)))

    
'Function to compute the invasion reproduction number in the generalisation of Alizon model defined in Eq. (13)'

def Alizon_RI(fixed_params, e, alpha2):
    Phi, nu, alpha1, e1 = fixed_params
    nu0 = nu1 = nu2 = nu11 = nu22 = nuc = nu
    e2 = e1
    ec = e
    alpha11 = alpha1
    alpha22 = alpha2
    alphac = (alpha11+alpha22)/2
    p1 = alpha11/(alpha11 + alpha22)
    
    # Defining the strain 1 endemic equilibrium
    m0estar = nu/alpha1
    m1estar = (1-e1)*(Phi-nu*m0estar)*m0estar/(Phi - alpha1*e1*m0estar**2)
    M1estar = Phi/nu - m0estar - m1estar
    
    # Defining the elements of the 2x2 submatrix (see Appendix C)
    A = m0estar*(alpha2*nuc + alphac*(1-p1)*(1-ec)*(alpha1*m1estar + alpha11*M1estar))/((nu2 + alpha1*m1estar + alpha11*M1estar)*nuc)
    B = m0estar*alphac*(1-p1)*(1-ec)/nuc
    D = (m1estar*alpha2*nuc + (m0estar*alphac*ec + m1estar*alphac*(1- p1))*(alpha1*m1estar + alpha11*M1estar))/((nu2 + alpha1*m1estar + alpha11*M1estar)*nuc)
    E = (m0estar*alphac*ec + m1estar*alphac*(1-p1))/nuc
    return max(1/2*(A+E + np.sqrt((A+E)**2-4*(A*E -D*B))), m0estar*alpha22*e2/nu22)
    

'Function to compute the invasion reproduction number in the two-slot model defined in Eq. (15)'

def twoslot_RI(fixed_params, e, alpha2):
    Phi, nu, alpha1, e1 = fixed_params
    nu0 = nu1 = nu2 = nu11 = nu22 = nuc = nu
    e2 = e1
    ec = e
    alpha11 = alpha1
    alpha22 = alpha2
    alphac = (alpha11+alpha22)/2
    p1 = alpha11/(alpha11 + alpha22)
    
    # Defining the strain 1 endemic equilibrium
    m0 = nu/alpha1    
    m1 = (1-e1)*(Phi-nu*m0)*m0/(Phi - alpha1*e1*m0**2)
    M1 = Phi/nu - m0 - m1
    
    # Defining the elements of the next generation matrix (see Appendix D)
    A = m0*(alpha2*nuc + alphac*(1-p1)*(1-ec)*(alpha1*m1+ alpha11*M1))/((nu2 + alpha1*m1 + alpha11*M1)*nuc)
    B = m0*alphac*(1-p1)*(1-ec)/nuc
    C = m0*alpha22*(1-e2)/nu22
    D = (m1*alpha2*nuc + (m0*alphac*2*p1*(1-p1)*ec + m1*alphac*(1-p1))*(alpha1*m1 + alpha11*M1))/((nu2 + alpha1*m1 + alpha11*M1)*nuc)
    E = (m0*alphac*2*p1*(1-p1)*ec + m1*alphac*(1-p1))/nuc
    F = alpha22*m1/nu22
    G = m0*alphac*(1-p1)**2*ec*(alpha1*m1 + alpha11*M1)/(nuc*(nu2 + alpha1*m1 + alpha11*M1))
    H = m0*alphac*(1-p1)**2*ec/nuc
    I = m0*alpha22*e2/nu22
    mat = np.array([[A, B, C], 
                    [D, E, F],
                    [G, H, I]])
    values,vectors=eig(mat)
    return max(values)

    
'Assigning the parameter values'

Phi = 2
nu = 10**-2
alpha1 = 10**-4

e1 = 0.5

RTT = alpha1*Phi/nu**2

print('RTT=', RTT)

'Plotting the invasion reproduction numbers for the different models with alpha2/alpha1 on y-axis and epsilon_c on x-axis'

# my_cmap_min and my_cmap_max are defined based on the min and max values of the invasion reproduction number computed for 
# each model
my_cmap_max = 2.5
my_cmap_min = 0.37
ColorSpectrum = np.logspace(np.log10(my_cmap_min),np.log10(my_cmap_max), 1000)

# Axis limits
# epsilon_c on the x-axis
X0 = 0
X1 = 1
# alpha2/alpha1 on the y-axis goes from 0.5 to 1.5
Y0 = 0.5
Y1 = 1.5

# Grid size
N = 50

# Grid values
epsilons = np.linspace(X0,X1,N+1)
alpha_ratios = np.linspace(Y0,Y1,N+1)
X,Y = np.meshgrid(epsilons,alpha_ratios)

fig, axes = plt.subplots(nrows=2, ncols=3)
axes[0,2].axis('off')

fixed_params = [Phi, nu, alpha1]

plt.subplot(231)
RIs = np.zeros((N+1,N+1))
for i, e in enumerate(epsilons):
    for j, alpha2 in enumerate(alpha_ratios*alpha1):
        RIs[j,i] = RI(fixed_params, e, alpha2)    
print('max =', np.max(RIs))
print('min =', np.min(RIs))
plt.xticks([])
plt.ylabel(r'$\alpha_2/\alpha_1$')
HM = plt.contourf(X,Y,RIs, levels=ColorSpectrum, cmap=my_cmap, norm = LogNorm(), extend='both')
plt.contour(X,Y,RIs,levels=[1],colors='black') # gives a black line where RI is 1
plt.text(0.1, 1.3, '(a)', fontsize = 'small')

plt.subplot(232)
RIs = np.zeros((N+1,N+1))
for i, e in enumerate(epsilons):
    for j, alpha2 in enumerate(alpha_ratios*alpha1):
        RIs[j,i] = normed_RI(fixed_params, e, alpha2)
print('max =', np.max(RIs))
print('min =', np.min(RIs))
plt.xticks([])
plt.yticks([])
HM = plt.contourf(X,Y,RIs, levels=ColorSpectrum, cmap=my_cmap, norm = LogNorm(), extend='both')
plt.contour(X,Y,RIs,levels=[1],colors='black') # gives a black line where RI is 1
plt.text(0.1, 1.3, '(a$^{\star}$)', fontsize = 'small') 

plt.subplot(234)
RIs = np.zeros((N+1,N+1))
for i, e in enumerate(epsilons):
    for j, alpha2 in enumerate(alpha_ratios*alpha1):
        RIs[j,i] = within_host_RI(fixed_params, e, alpha2)
print('max =', np.max(RIs))
print('min =', np.min(RIs))
plt.xlabel(r'$\epsilon_c$')
plt.xticks([0, 0.5, 1], ['0', '0.5', '1'])
plt.ylabel(r'$\alpha_2/\alpha_1$')
HM = plt.contourf(X,Y,RIs, levels=ColorSpectrum, cmap=my_cmap, norm = LogNorm(), extend='both')
plt.contour(X,Y,RIs,levels=[1],colors='black') # gives a black line where RI is 1
plt.text(0.1, 1.3, '(b)', fontsize = 'small')


fixed_params = [Phi, nu, alpha1, e1]

plt.subplot(235)
RIs = np.zeros((N+1,N+1))
for i, e in enumerate(epsilons):
    for j, alpha2 in enumerate(alpha_ratios*alpha1):
        RIs[j,i] = Alizon_RI(fixed_params, e, alpha2)
print('max =', np.max(RIs))
print('min =', np.min(RIs))
plt.xlabel(r'$\epsilon_c$')
plt.xticks([0, 0.5, 1], ['0', '0.5', '1'])
plt.yticks([])
HM = plt.contourf(X,Y,RIs, levels = ColorSpectrum, cmap=my_cmap, norm = LogNorm(), extend='both')
plt.contour(X,Y,RIs,levels=[1],colors='black') # gives a black line where RI is 1
plt.text(0.1, 1.3, '(c)', fontsize = 'small')

plt.subplot(236)
RIs = np.zeros((N+1,N+1))
for i, e in enumerate(epsilons):
    for j, alpha2 in enumerate(alpha_ratios*alpha1):
        RIs[j,i] = twoslot_RI(fixed_params, e, alpha2)
print('max =', np.max(RIs))
print('min =', np.min(RIs))
plt.xlabel(r'$\epsilon_c$')
plt.xticks([0, 0.5, 1], ['0', '0.5', '1'])
plt.yticks([])
HM = plt.contourf(X,Y,RIs, levels=ColorSpectrum, cmap=my_cmap, norm = LogNorm(), extend='both')
plt.contour(X,Y,RIs,levels=[1],colors='black') # gives a black line where RI is 1
plt.text(0.1, 1.3, '(d)', fontsize = 'small')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(HM, label='$R_I$',ticks=[0.4, 1, 2.5], cax=cbar_ax)

plt.savefig('RI_heatmaps.png', bbox_inches = 'tight')