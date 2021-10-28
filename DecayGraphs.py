#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from DES import Differential_Equation_Solver_new as DES


# In[2]:


alpha = 1/137         
dtda_part2 = 2*np.pi/3     
f_pi = 131              
Gf = 1.166*10**-11      
me = .511               
mpi_neutral = 135       
mpi_charged = 139.569   
mPL = 1.124*10**22      
mu = 105.661  
eps_e = me/mu
Enumax = (mu/2)*(1-(eps_e)**2)


# In[3]:


n = 10                       
num = 128                      
steps = 120                   
d_steps = 1                   
a_init = 1/50                 
a_final = 2                   
y_init = np.zeros(num) 
y_init[-2] = 50               
y_init[-1] = 0                
boxsize = 2
e_array = np.linspace(0,int(num-2)*boxsize,int(num-2))     
x_values, w_values = np.polynomial.laguerre.laggauss(10) 
x_valuese, w_valuese = np.polynomial.legendre.leggauss(10)

D = 1            
fT = .5          


# In[4]:


def I1(xi,x): 
    numerator = (np.e**xi)*(xi**2)*((xi**2+x**2)**.5)
    denominator = np.e**((xi**2+x**2)**.5)+1
    E_den = numerator/denominator
    return E_den

def I2(xi,x):
    numerator = (np.e**xi)*(xi**4)
    denominator = ((xi**2+x**2)**.5)*(np.e**((xi**2+x**2)**.5)+1)
    Pressure = numerator/denominator
    return Pressure

def dI1(xi,x): 
    numerator = (np.e**xi)*((xi**2+x**2)**.5)
    denominator = np.e**((xi**2+x**2)**.5)+1
    dE_den = numerator/denominator
    return dE_den

def dI2(xi,x):
    numerator = (np.e**xi)*3*(xi**2)
    denominator = ((xi**2+x**2)**.5)*(np.e**((xi**2+x**2)**.5)+1)
    dPressure = numerator/denominator
    return dPressure

def calculate_integral(I,x): 
    integral = np.sum(w_values*I(x_values,x)) 
    return integral

def trapezoid(array,dx): #dx will just be boxsize for our cases currently
    total = np.sum(dx*(array[1:]+array[:-1])/2)
    return total


# In[5]:


def rate1(ms,mixangle): 
    numerator = 9*(Gf**2)*alpha*(ms**5)*((np.sin(mixangle))**2)
    denominator = 512*np.pi**4
    Gamma = numerator/denominator
    return Gamma

def rate2(ms,mixangle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    part2 = ms*((ms**2)-(mpi_neutral**2))*(np.sin(mixangle))**2
    Gamma = part1*part2
    return Gamma

def rate3(ms,mixangle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    parentheses = ((ms**2) - (mpi_charged+me)**2)*((ms**2) - (mpi_charged-me)**2)
    part2 = ms * ((parentheses)**(1/2)) * (np.sin(mixangle))**2
    Gamma = part1*part2
    return 2*Gamma

def rate4(ms,mixangle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    parentheses = ((ms**2) - (mpi_charged+mu)**2)*((ms**2) - (mpi_charged-mu)**2)
    part2 = ms * ((parentheses)**(1/2)) * (np.sin(mixangle))**2
    Gamma = part1*part2
    return 2*Gamma 

def rate5(ms,mixangle):
    numerator = (Gf**2)*(ms**5)*((np.sin(mixangle))**2)
    denominator = 192*np.pi**3
    Gamma = numerator/denominator
    return Gamma

def rate6(ms,mixangle):
    Gamma = (1/3)*(rate5(ms,mixangle))
    return Gamma

def rate8(ms,mixangle):
    Gamma = (1/3)*(rate5(ms,mixangle))
    return Gamma

def ts(ms,mixangle):
    tau = 1/(rate1(ms,mixangle)+rate2(ms,mixangle)+rate3(ms,mixangle)+rate4(ms,mixangle))
    return tau

def ts_expanded(ms,mixangle):
    tau_inv_MeV = 1/(rate1(ms,mixangle)+rate2(ms,mixangle)+rate3(ms,mixangle)+rate4(ms,mixangle)+rate5(ms,mixangle)+rate6(ms,mixangle)+rate8(ms,mixangle))
    tau_s = tau_inv_MeV*(6.582*10**-22)
    return tau_s

def decayrates(ms,mixangle):
    Gamma1= Gamma2= Gamma3= Gamma4= Gamma5= Gamma6= Gamma8 = 0
    if (ms>0):
        Gamma1 = rate1(ms,mixangle)
        Gamma5 = rate5(ms,mixangle) 
    if (ms>1.022): #the mass of two electrons
        Gamma6 = rate6(ms,mixangle)
    if (ms>211.32): #the mass of two muons
        Gamma8 = rate8(ms,mixangle)
    if (ms>mpi_neutral): #mass of a neutral pion
        Gamma2 = rate2(ms,mixangle)
    if (ms>(mpi_charged+me)): #mass of a charged pion plus the mass of an electron
        Gamma3 = rate3(ms,mixangle)
    if (ms>(mpi_charged+mu)): #mass of a charged pion plus the mass of a muon
        Gamma4 = rate4(ms,mixangle)
    return Gamma1, Gamma2, Gamma3, Gamma4, Gamma5, Gamma6, Gamma8

def branching_ratios(ms,mixangle):
    Gamma1, Gamma2, Gamma3, Gamma4, Gamma5, Gamma6, Gamma8 = decayrates(ms,mixangle)
    total = Gamma1 + Gamma2 + Gamma3 + Gamma4 + Gamma5 + Gamma6 + Gamma8
    ratio1 = Gamma1/total
    ratio2 = Gamma2/total
    ratio3 = Gamma3/total
    ratio4 = Gamma4/total
    ratio5 = Gamma5/total
    ratio6 = Gamma6/total
    ratio8 = Gamma8/total
    return ratio1,ratio2,ratio3,ratio4,ratio5,ratio6,ratio8


# In[6]:


def diracdelta(E,E_I,de): 
    difference = E-E_I
    if -(1/2)*de<= difference <(1/2)*de: 
        output = 1/de
    else:
        output = 0
    return output
    
def diracdelta2(E,EBmin,EBmax,E_B,gamma,v):  # hmm, doesn't send boxsize
    if (EBmin-boxsize)<=E<=EBmin: #if energy is in the first box
        tophat = (E+boxsize-EBmin)/(2*gamma*v*E_B*boxsize)
    elif (EBmax-boxsize)<=E<=EBmax: #if energy is in the last box
        tophat = (EBmax-E)/(2*gamma*v*E_B*boxsize)
    elif EBmin<= E <=EBmax:  #if energy is in any of the middle boxes 
        tophat = 1/(2*gamma*v*E_B) 
    else:
        tophat = 0
    return tophat

def EB(mA,mB,mC): 
    E_B = (mA**2 + mB**2 - mC**2)/(2*mA)
    return E_B

def Gammamua(a,b): #for both electron neutrinos and muon neutrinos for decay types III and IV
    if a>Enumax:
        return 0
    constant = 8*Gf*(mu**2)/(16*np.pi**3)
    part_b1 = (-1/4)*(me**4)*mu*np.log(abs(2*b-mu))
    part_b2 = (-1/6)*b
    part_b3 = 3*(me**4)+6*(me**2)*mu*b
    part_b4 = (mu**2)*b*(4*b-3*mu)
    part_b = (part_b1+part_b2*(part_b3+part_b4))/mu**3
    part_a1 = (-1/4)*(me**4)*mu*np.log(abs(2*a-mu))
    part_a2 = (-1/6)*a
    part_a3 = 3*(me**4)+6*(me**2)*mu*a
    part_a4 = (mu**2)*a*(4*a-3*mu)
    part_a = (part_a1+part_a2*(part_a3+part_a4))/mu**3
    integral = part_b-part_a
    Gam_mua = constant*integral
    return Gam_mua

def Gammamub(): #for both electron neutrinos and muon neutrinos for decay types III and IV 
    constant = 8*Gf*(mu**2)/(16*np.pi**3)
    part_a1 = 3*(me**4)*(mu**2)*np.log(abs(2*Enumax-mu))
    part_a2 = 6*(me**4)*Enumax*(mu+Enumax)
    part_a3 = 16*(me**2)*mu*(Enumax**3)
    part_a4 = 4*(mu**2)*(Enumax**3)*(3*Enumax - 2*mu)
    part_a5 = 24*mu**3
    part_b1 = 3*(me**4)*(mu**2)*np.log(abs(-mu))/part_a5
    integral = ((part_a1+part_a2+part_a3+part_a4)/part_a5)-part_b1
    Gam_mub = -1*constant*integral
    return Gam_mub

Gam_mub = Gammamub()

def u_integral(E_mumin,E_mumax,Eactive):
    Eu_array = ((E_mumax-E_mumin)/2)*x_valuese + ((E_mumax+E_mumin)/2)
    integral = 0
    for i in range(n):
        gammau = Eu_array[i]/mu
        pu = (Eu_array[i]**2-mu**2)**(1/2)
        vu = pu/Eu_array[i]
        Gam_mua = Gammamua(Eactive/(gammau*(1+vu)),min(Enumax,Eactive/(gammau*(1-vu))))
        integral = integral + (w_valuese[i]*((E_mumax-E_mumin)/2)*(1/(2*gammau*vu))*Gam_mua)
    return integral


# In[7]:


def driver1(ms,mixangle,ns):

    E_B1 = ms/2 
    
    def derivatives1(a,y): 
        d_array1 = np.zeros(len(y))
        Tcm = y_init[-2]*a_init/a
        
        dtda_part1 = mPL/(2*a)
        dtda_part3 = (y[-2]**4*np.pi**2)/15
        dtda_part4 = 2*y[-2]**4*calculate_integral(I1,me/y[-2])/np.pi**2
        dtda_part5 = (7*(np.pi**2)*(Tcm**4))/40
        dtda_part6 = ms*ns(Tcm,y[-1],ms,mixangle)
        dtda_part7 = trapezoid(y[1:int(num-2)]/(e_array[1:int(num-2)]**2*a**3),boxsize) #index 1 is starting point to avoid dividing by 0
        dtda = dtda_part1/(dtda_part2*(dtda_part3+dtda_part4+dtda_part5+dtda_part6+dtda_part7))**.5
        d_array1[-1] = dtda
        
        dTda_constant1 = (4*np.pi**2/45)+(2/np.pi**2)*(calculate_integral(I1,me/y[-2]) + (1/3)*(calculate_integral(I2,me/y[-2])))
        dTda_constant2 = 2*me**2*a**3/(np.pi**2)
        dTda_numerator1 = -3*a**2*y[-2]**3*dTda_constant1
        dTda_numerator2 = ((fT*ms*a**3)/(ts(ms,mixangle)*y[-2]))*ns(Tcm,y[-1],ms,mixangle)*dtda
        dTda_denominator = (3*y[-2]**2*a**3*dTda_constant1) + dTda_constant2*(calculate_integral(dI1,me/y[-2]) + (1/3)*(calculate_integral(dI2,me/y[-2])))
        dTda = (dTda_numerator1 + dTda_numerator2)/dTda_denominator
        d_array1[-2] = dTda
    
        for i in range (1,num-2):
            dPdtdE1 = rate1(ms,mixangle)*diracdelta((i*boxsize),E_B1,boxsize)
            d_array1[i] = dPdtdE1*ns(Tcm,y[-1],ms,mixangle)*a**3*dtda
            
        return d_array1
    a_array1, y_matrix1 = DES.destination_x(derivatives1, y_init, steps, d_steps, a_init, a_final)
    return a_array1,y_matrix1


# In[8]:


def driver2(ms,mixangle,ns):
    
    E_B2 = (ms**2 - mpi_neutral**2)/(2*ms)
    
    def derivatives2(a,y):
        d_array2 = np.zeros(len(y))
        Tcm = y_init[-2]*a_init/a
        
        dtda_part1 = mPL/(2*a)
        dtda_part3 = (y[-2]**4*np.pi**2)/15
        dtda_part4 = 2*y[-2]**4*calculate_integral(I1,me/y[-2])/np.pi**2
        dtda_part5 = (7*(np.pi**2)*(Tcm**4))/40
        dtda_part6 = ms*ns(Tcm,y[-1],ms,mixangle)
        dtda_part7 = trapezoid(y[1:int(num-2)]/(e_array[1:int(num-2)]**2*a**3),boxsize) #1 is included to avoid dividing by 0
        dtda = dtda_part1/(dtda_part2*(dtda_part3+dtda_part4+dtda_part5+dtda_part6+dtda_part7))**.5
        d_array2[-1] = dtda
        
        dTda_constant1 = (4*np.pi**2/45)+(2/np.pi**2)*(calculate_integral(I1,me/y[-2]) + (1/3)*(calculate_integral(I2,me/y[-2])))
        dTda_constant2 = 2*me**2*a**3/(np.pi**2)
        dTda_numerator1 = -3*a**2*y[-2]**3*dTda_constant1
        dTda_numerator2 = ((fT*ms*a**3)/(ts(ms,mixangle)*y[-2]))*ns(Tcm,y[-1],ms,mixangle)*dtda
        dTda_denominator = (3*y[-2]**2*a**3*dTda_constant1) + dTda_constant2*(calculate_integral(I1,me/y[-2]) + (1/3)*(calculate_integral(dI2,me/y[-2])))
        dTda = (dTda_numerator1 + dTda_numerator2)/dTda_denominator
        d_array2[-2] = dTda
    
        for i in range (1,num-2):
            dPdtdE2 = rate2(ms,mixangle)*diracdelta((i*boxsize),E_B2,boxsize)
            d_array2[i] = dPdtdE2*ns(Tcm,y[-1],ms,mixangle)*a**3*dtda
              
        return d_array2
    a_array2, y_matrix2 = DES.destination_x(derivatives2, y_init, steps, d_steps, a_init, a_final)
    return a_array2,y_matrix2


# In[9]:


def driver3(ms,mixangle,ns):
    
    #constants referring to initial part of decay 3:
    E_pi3 = EB(ms,mpi_charged,me) 
    p_pi3 = (E_pi3**2 - mpi_charged**2)**(1/2) 
    v3 = p_pi3/E_pi3
    gammapi3 = E_pi3/mpi_charged
    
    #constants referring to decay 3a:
    E_B3 = EB(mpi_charged,0,mu)
    E_B3max = gammapi3*E_B3*(1+v3)
    E_B3min = gammapi3*E_B3*(1-v3)
                        
    #additional constants referring to decay 3b:
    E_mu3 = EB(mpi_charged,mu,0) 
    p_mu3 = (E_mu3**2 - mu**2)**(1/2) 
    E_mumax3 = gammapi3*(E_mu3 + (v3*p_mu3))
    E_mumin3 = gammapi3*(E_mu3 - (v3*p_mu3))
                        
    def derivatives3(a,y): 
        d_array3 = np.zeros(len(y))
        Tcm = y_init[-2]*a_init/a
        
        dtda_part1 = mPL/(2*a)
        dtda_part3 = (y[-2]**4*np.pi**2)/15
        dtda_part4 = 2*y[-2]**4*calculate_integral(I1,me/y[-2])/np.pi**2
        dtda_part5 = (7*(np.pi**2)*(Tcm**4))/40
        dtda_part6 = ms*ns(Tcm,y[-1],ms,mixangle)
        dtda_part7 = trapezoid(y[1:int(num-2)]/(e_array[1:int(num-2)]**2*a**3),boxsize) #1 is included to avoid dividing by 0
        dtda = dtda_part1/(dtda_part2*(dtda_part3+dtda_part4+dtda_part5+dtda_part6+dtda_part7))**.5
        d_array3[-1] = dtda
        
        dTda_constant1 = (4*np.pi**2/45)+(2/np.pi**2)*(calculate_integral(I1,me/y[-2]) + (1/3)*(calculate_integral(I2,me/y[-2])))
        dTda_constant2 = 2*me**2*a**3/(np.pi**2)
        dTda_numerator1 = -3*a**2*y[-2]**3*dTda_constant1
        dTda_numerator2 = ((fT*ms*a**3)/(ts(ms,mixangle)*y[-2]))*ns(Tcm,y[-1],ms,mixangle)*dtda
        dTda_denominator = (3*y[-2]**2*a**3*dTda_constant1) + dTda_constant2*(calculate_integral(dI1,me/y[-2]) + (1/3)*(calculate_integral(dI2,me/y[-2])))
        dTda = (dTda_numerator1 + dTda_numerator2)/dTda_denominator
        d_array3[-2] = dTda
    
        for i in range (1,num-2):
            dPdtdE3a = .5*rate3(ms,mixangle)*diracdelta2((i*boxsize),E_B3min,E_B3max,E_B3,gammapi3,v3)
            dPdtdE3b = (rate3(ms,mixangle)/(2*gammapi3*v3*p_mu3*Gam_mub))*u_integral(E_mumin3,E_mumax3,i*boxsize)
            d_array3[i] = (dPdtdE3a+dPdtdE3b)*ns(Tcm,y[-1],ms,mixangle)*a**3*dtda #all the neutrinos (antineutrinos NOT included; just scale d6s and d7s differently if you want the antineutrinos)
              
        return d_array3
    a_array3, y_matrix3 = DES.destination_x(derivatives3, y_init, steps, d_steps, a_init, a_final)
    return a_array3,y_matrix3


# In[10]:


def driver4(ms,mixangle,ns):
    
    #constants referring the initial decay of decay 4
    E_pi4 = EB(ms,mpi_charged,mu) 
    p_pi4 = (E_pi4**2 - mpi_charged**2)**(1/2)
    v4 = p_pi4/E_pi4
    gammapi4 = E_pi4/mpi_charged
    Eu = ms-E_pi4 
    
    #constants referring to decay 4b:
    E_B4 = EB(mpi_charged,0,mu)
    E_B4max = gammapi4*E_B4*(1 + v4)
    E_B4min = gammapi4*E_B4*(1 - v4)
    
    #constants referring to decay 4c:
    E_mu4 = EB(mpi_charged,mu,0)
    p_mu4 = (E_mu4**2 - mu**2)**(1/2) 
    E_mumax4 = gammapi4*(E_mu4 + (v4*p_mu4))
    E_mumin4 = gammapi4*(E_mu4 - (v4*p_mu4))
 
    def derivatives4(a,y):
        d_array4 = np.zeros(len(y))
        Tcm = y_init[-2]*a_init/a
        
        dtda_part1 = mPL/(2*a)
        dtda_part3 = (y[-2]**4*np.pi**2)/15
        dtda_part4 = 2*y[-2]**4*calculate_integral(I1,me/y[-2])/np.pi**2
        dtda_part5 = (7*(np.pi**2)*(Tcm**4))/40
        dtda_part6 = ms*ns(Tcm,y[-1],ms,mixangle)
        dtda_part7 = trapezoid(y[1:int(num-2)]/(e_array[1:int(num-2)]**2*a**3),boxsize) #1 is included to avoid dividing by 0
        dtda = dtda_part1/(dtda_part2*(dtda_part3+dtda_part4+dtda_part5+dtda_part6+dtda_part7))**.5
        d_array4[-1] = dtda
        
        dTda_constant1 = (4*np.pi**2/45)+(2/np.pi**2)*(calculate_integral(I1,me/y[-2]) + (1/3)*(calculate_integral(I2,me/y[-2])))
        dTda_constant2 = 2*me**2*a**3/(np.pi**2)
        dTda_numerator1 = -3*a**2*y[-2]**3*dTda_constant1
        dTda_numerator2 = ((fT*ms*a**3)/(ts(ms,mixangle)*y[-2]))*ns(Tcm,y[-1],ms,mixangle)*dtda
        dTda_denominator = (3*y[-2]**2*a**3*dTda_constant1) + dTda_constant2*(calculate_integral(dI1,me/y[-2]) + (1/3)*(calculate_integral(dI2,me/y[-2])))
        dTda = (dTda_numerator1 + dTda_numerator2)/dTda_denominator
        d_array4[-2] = dTda
    
        for i in range (1,num-2):
            Gam_mua = Gammamua((i*boxsize)/(gammapi4*(1+v4)),min(Enumax,(i*boxsize)/(gammapi4*(1-v4))))
            dPdtdE4a = rate4(ms,mixangle)*(1/(2*gammapi4*v4))*(Gam_mua/Gam_mub)
            dPdtdE4b = .5*rate4(ms,mixangle)*diracdelta2((i*boxsize),E_B4min,E_B4max,E_B4,gammapi4,v4)
            dPdtdE4c = rate4(ms,mixangle)*(1/(2*gammapi4*v4*p_mu4*Gam_mub))*u_integral(E_mumin4,E_mumax4,i*boxsize)
            d_array4[i] = (dPdtdE4a+dPdtdE4b+dPdtdE4c)*ns(Tcm,y[-1],ms,mixangle)*a**3*dtda 
              
        return d_array4
    a_array4, y_matrix4 = DES.destination_x(derivatives4, y_init, steps, d_steps, a_init, a_final)
    return a_array4,y_matrix4


# In[11]:


def calc_final_num(ms,mixangle):
    
    def ns(Tcm,t,ms,mixangle): 
        part1 = D*3*1.20206/(2*np.pi**2)
        part2 = Tcm**3*np.e**(-t/ts(ms,mixangle))
        n_s = part1*part2
        return n_s
    
    a_array1,y_matrix1 = driver1(ms,mixangle,ns)
    a_array2,y_matrix2 = np.zeros(np.shape(a_array1)), np.zeros(np.shape(y_matrix1))
    a_array3,y_matrix3 = np.zeros(np.shape(a_array1)), np.zeros(np.shape(y_matrix1))
    a_array4,y_matrix4 = np.zeros(np.shape(a_array1)), np.zeros(np.shape(y_matrix1))
    if (ms>mpi_neutral): 
        a_array2,y_matrix2 = driver2(ms,mixangle,ns)
    if (ms>(mpi_charged+me)): 
        a_array3,y_matrix3 = driver3(ms,mixangle,ns)
    if (ms>(mpi_charged+mu)): 
        a_array4,y_matrix4 = driver4(ms,mixangle,ns)
    
    num1 = np.zeros(len(e_array)) 
    num2 = np.zeros(len(e_array)) 
    num3 = np.zeros(len(e_array)) 
    num4 = np.zeros(len(e_array)) 
    numsum = np.zeros(len(e_array)) 
    
    for i in range (len(e_array)):
        num1[i] = y_matrix1[i][-1]
        num2[i] = y_matrix2[i][-1]
        num3[i] = y_matrix3[i][-1]
        num4[i] = y_matrix4[i][-1]
        numsum[i] = num1[i] + num2[i] + num3[i] + num4[i]
    
    return num1, num2, num3, num4, numsum

def graph_final_num(ms,mixangle):
    totalnum1, totalnum2, totalnum3, totalnum4, totalnumsum = calc_final_num(ms,mixangle)
    plt.figure(figsize=(10,10))
    plt.plot(e_array,totalnum1,color = 'gold', label="$\\nu_s \\to \\nu\gamma$")
    plt.plot(e_array,totalnum2,color = 'salmon', linestyle =":", label="$\\nu_s \\to \\nu\pi^0$")
    plt.plot(e_array,totalnum3,color = 'palevioletred', linestyle ="--", label="$\\nu_s \\to \pi^{\pm}e^{\mp}$")
    plt.plot(e_array,totalnum4,color = 'rebeccapurple', linestyle ="-.", label="$\\nu_s \\to \pi^{\pm}\mu^{\mp}$")
    plt.plot(e_array,totalnumsum, color='black',label='Total')
    plt.xlabel("Energy of Active Neutrino (MeV)",fontsize=20)
    plt.ylabel("Number Density (arb. units)",fontsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.xlim(0,11*ms/20)
    plt.legend(loc="upper left",fontsize=20)
    plt.savefig("NumberDensity.pdf")


# In[12]:


def graph_BR(mixangle):
    mass_arr = np.linspace(50,500,10000)
    process1 = np.zeros(len(mass_arr))
    process2 = np.zeros(len(mass_arr))
    process3 = np.zeros(len(mass_arr))
    process4 = np.zeros(len(mass_arr))
    process5 = np.zeros(len(mass_arr))
    process6 = np.zeros(len(mass_arr))
    process8 = np.zeros(len(mass_arr))
    for i in range (len(mass_arr)):
        process1[i], process2[i], process3[i], process4[i], process5[i], process6[i], process8[i] = branching_ratios(mass_arr[i],mixangle)
    
    plt.figure(figsize=(10,10))
    plt.plot(mass_arr,process1,color = 'gold', label="$\\nu_s \\to \\nu\gamma$")
    plt.plot(mass_arr,process2,color = 'salmon', linestyle =":", label="$\\nu_s \\to \\nu\pi^0$")
    plt.plot(mass_arr,process3,color = 'palevioletred', linestyle ="--", label="$\\nu_s \\to \pi^{\pm}e^{\mp}$")
    plt.plot(mass_arr,process4,color = 'rebeccapurple', linestyle ="-.", label="$\\nu_s \\to \pi^{\pm}\mu^{\mp}$")
    plt.plot(mass_arr,process5+process6+process8, color = 'indigo', label="Others")
    plt.tick_params(axis="x", labelsize=20)
    plt.tick_params(axis="y", labelsize=20)
    plt.xlabel("Mass (MeV)", fontsize = 20)
    plt.xticks(np.linspace(50,500,10))
    plt.ylabel("Branching Ratio", fontsize = 20)
    plt.legend(loc=0, fontsize=20)
    plt.savefig("Branchingratio.pdf")
    
def graph_mix_ms_t(ms1,ms2,mix1,mix2):
    n = 100
    mass_arr = np.linspace(ms1,ms2,n)
    mixangle_arr = np.linspace(mix1,mix2,n)
    tau_matrix = np.zeros((n,n))

    for i in range (n):
        for j in range (n):
            tau_matrix[i][j] = ts_expanded(mass_arr[i],mixangle_arr[j])
        
    time_array = [0.01,0.02,0.03,0.05,0.1,0.25,0.5,1.0,2.0]
    colors = [cm.magma(x) for x in np.linspace(0.05, 0.8, len(time_array))]

    fig1, ax1 = plt.subplots(figsize=(10,10))
    CS1 = ax1.contour(mixangle_arr*10**5, mass_arr, tau_matrix, time_array, colors = colors)
    fmt = {}
    strs = ['0.01 s','0.02 s','0.03 s','0.05 s','0.1 s','0.25 s','0.5 s','1.0 s','2.0 s']
    for l, s in zip(CS1.levels, strs):
           fmt[l] = s

    ax1.clabel(CS1, fmt=fmt, fontsize=18)
    ax1.tick_params(axis="x", labelsize=20)
    ax1.tick_params(axis="y", labelsize=20)
    plt.xticks(np.linspace(2,10,5))
    plt.xlabel("Mixing Angle ($\\times 10^{-5}$ Radians)", fontsize=20)
    ax1.set_ylabel("Mass (MeV)", fontsize=20)
    plt.savefig("DecayTimes.pdf")

