# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:25:11 2024

@author: robbi
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.signal import argrelextrema
from numpy.fft import fftfreq
import multiprocessing
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 
import csv
from scipy.special import laguerre
from scipy.fft import fft
from scipy import correlate

N = 450 # number of harmonic oscillator basis states
a = destroy(N)                  #harmonic oscillator lowering operator
adag = create(N)  
omega_0 = 1
Omega = 1. * omega_0  # qubit frequency
opts = Options(nsteps=10000,store_states=True) 

Hq_sc = tensor(sigmax(),qeye(N))
Hq_sc_p = tensor(sigmap(),qeye(N))
Hq_sc_m = tensor(sigmam(),qeye(N))
Hq_a_p = tensor(sigmap(),a)
Hq_adag_m = tensor(sigmam(),adag)
Hq_a_m = tensor(sigmam(),a)
Hq_adag_p = tensor(sigmap(),adag)
Hq_spin = Omega/2*tensor(sigmaz(),qeye(N))
sigma_z = tensor(sigmaz(),qeye(N))
norm_const = 1/np.sqrt(2)
psi_plusz = basis(2,0)       #sigma_z eigenstates for the spin
psi_minusz = basis(2,1)
psi_had_plus = norm_const*(basis(2,0)+basis(2,1))
psi_y_plus = norm_const*(basis(2,0)+1j*basis(2,1))


#def purity(rho):
    #return np.trace(rho**2)
    
def alpha_inversion(eigs):
    eigenvalues = []
    for eigvals in eigs:
        diff = eigvals[0] - eigvals[1]
        eigenvalues.append(diff)
    return eigenvalues
        
    
def shannon(states):
    shannon_entropy = []
    for state in states:
        print(np.trace(state.full()))
        p = np.diag(state.full())
        p_nonzero = p[p>0]
        summation = -1*np.sum(p_nonzero*np.log(p_nonzero))
        shannon_entropy.append(summation)
    return shannon_entropy

def bloch(states):
    b = Bloch()

    for state in states:
        b.add_states(state,kind='point')
        b.show()
        
        


def compute(params):
    
    gvals, Avals = params
    
    entropy_q = []
    entropy_qRWA = []
    purity_q = []
    purity_q_2 = []
    avg_purity_q = []
    purity_qRWA = []
    atomic_inversion_q = []
    sigma_p_q = []
    sigma_m_q = []
    atomic_inversion_rwa = []
    
    for j in range(len(Avals)):
        A = Avals[j]
        
        args_sc = {'A' : A, 'omega_0' : omega_0}
    
        def Hq_sc_coeff(t, args):
            A0, omega = args_sc['A'], args_sc['omega_0']
            return A0*2*np.cos(omega * t)
        
        def Hq_sc_p_coeff(t, args):
            A0, omega = args_sc['A'], args_sc['omega_0']
            return A0*np.exp(-1j*omega * t)
        
        def Hq_sc_m_coeff(t, args):
            A0, omega = args_sc['A'], args_sc['omega_0']
            return A0*np.exp(1j*omega * t)
    
        # DEFINE THE TIMESCALE AND FREQ ARRAY FOR FFT
        sc_rabi_freq = A/np.pi
        sc_rabi_T = 1/sc_rabi_freq
        
        
        H_sc = [Hq_spin, [Hq_sc, Hq_sc_coeff]]
        H_scRWA = [Hq_spin, [Hq_sc_p, Hq_sc_p_coeff], [Hq_sc_m, Hq_sc_m_coeff]]
        
        '''SEMICLASSICAL HAMILTONIAN SOLUTIONS'''
        
        #result_sc = mesolve(H_sc, psi0, tlist, [], [], options=opts)
        #psi_sc = [psi.ptrace(1) for psi in result_sc.states] 
        #result_scRWA = mesolve(H_scRWA, psi0, tlist, [], [], options=opts)
        #prob_e_sc = [np.real_if_close(psi.ptrace(0)[0,0]) for psi in result_sc.states]
        #prob_e_scRWA = [np.real_if_close(psi.ptrace(0)[0,0]) for psi in result_scRWA.states]
        
        for i in range(len(gvals)):
            g = gvals[i]
            alpha = A/g
            tlist = np.linspace( 0,1/g,5000)
            disp = displace(N,alpha).dag()
            cat_state = (fock(N,0)+(displace(N,2*alpha).dag()*fock(N,0))).unit()
            psi0 = tensor(psi_plusz,fock(N,0))
            result_sc = mesolve(H_sc, psi0, tlist, [], [], options=opts)
            states_sc = [psi.ptrace(0) for psi in result_sc.states]
            shann_sc = shannon(states_sc)
            #initial state
            args_q = {'g' : g, 'omega_0' : omega_0, 'alpha': alpha}
        
            
            def Hq_a_coeff(t, args):
                g0, omega = args_q['g'], args_q['omega_0']
                return g0*np.exp(-1j*omega * t)
            
            def Hq_adag_coeff(t, args):
                g0, omega = args_q['g'], args_q['omega_0']
                return g0*np.exp(1j*omega * t)
            
            H_q = [Hq_spin, [Hq_sc, Hq_sc_coeff], [Hq_a_p, Hq_a_coeff], [Hq_adag_m, Hq_adag_coeff], [Hq_a_m, Hq_a_coeff], [Hq_adag_p, Hq_adag_coeff]]
            H_qRWA = [Hq_spin, [Hq_sc_p, Hq_sc_p_coeff], [Hq_sc_m, Hq_sc_m_coeff], [Hq_a_p, Hq_a_coeff], [Hq_adag_m, Hq_adag_coeff]]
            
            '''QUANTUM HAMILTONIAN SOLUTIONS'''                                            
            result_q = mesolve(H_qRWA, psi0, tlist, [], e_ops=[sigma_z,Hq_sc_p,Hq_sc_m], options=opts) 
            
           
            SvN = [entropy_vn(psi.ptrace(0)) for psi in result_q.states]
            #pur_q = [entropy_vn(psi.ptrace(1)) for psi in result_q.states]
            #pur_q_2 = [entropy_vn(psi.ptrace(0)) for psi in result_q.states]
            
            probN = [psi.ptrace(1).diag()[N-1] for psi in result_q.states]
            #probN_2 = [psi.ptrace(0).diag()[-1] for psi in result_q.states]
            print(g)
            if np.max(probN)>0.001:
                print(g)
                print("WARNING: Consider increasing Hilbert space size")
                
            #avg_pur_q = np.mean(pur_q)
            atom_i = [exp for exp in result_q.expect[0]]
            sigma_p = [exp for exp in result_q.expect[1]]
            sigma_m = [exp for exp in result_q.expect[2]]
            #atom_i_rwa = [exp for exp in result_qRWA.expect[0]]
            #SvN_RWA = [entropy_vn(psi.ptrace(0)) for psi in result_qRWA.states]
            
            q_state_atom = [psi.ptrace(0) for psi in result_q.states]
            q_state = result_q.states
            eigs = [psi.ptrace(0).eigenenergies() for psi in result_q.states]
            eigs = alpha_inversion(eigs)
            entropy_q.append(SvN)
            #entropy_qRWA.append(SvN_RWA)
            #purity_q.append(pur_q)
            #purity_q_2.append(pur_q_2)
            #avg_purity_q.append(avg_pur_q)
            atomic_inversion_q.append(atom_i)
            sigma_p_q.append(sigma_p)
            sigma_m_q.append(sigma_m)
            #atomic_inversion_rwa.append(atom_i_rwa)
            
    return q_state,eigs,shann_sc,shannon(q_state),sigma_p_q,sigma_m_q,entropy_q,atomic_inversion_q,tlist


def save_data(file_name, data, A_vals, g_values):
    # Initialize an empty array with extra space for g_values in the first column and A_vals in the first row
    full_data = np.zeros((len(g_values) + 1, len(A_vals) + 1))
    
    # Set the first row (after the first cell) to A_vals
    full_data[0, 1:] = A_vals
    # Set the first column (after the first cell) to g_values
    full_data[1:, 0] = g_values
    # Fill in the rest of the array with the data
    full_data[1:, 1:] = data.T  # Transpose data to align with g_values and A_vals

    # Save the full_data array to a file, formatting numbers for readability
    # Use '%-10.5f' to format the numbers with 5 decimal places, left-justified in a 10 character field
    np.savetxt(file_name, full_data, delimiter=' ', fmt='%-10.5f', header=' '.join(['       '] + [f'{a:10.5f}' for a in A_vals]), comments='')


def main():
    gvals_log =np.array([0.001])
    A_vals = np.array([0.01])
    parameters = (gvals_log, A_vals)  # Pass parameters directly, not in a list
    
    # Since we're only computing one set of parameters, no need for multiprocessing
    states,eigs,shann_sc,shann,sigma_p_q,sigma_m_q,entropy_q,atomic_inversion_q, time = compute(parameters)
    
    qsave(states,'+z_collapse_states_RWA_N450')
    

    #purity_q = np.array(purity_q)
    #purity_q_2 = np.array(purity_q_2)
    atomic_inversion_q = np.array(atomic_inversion_q)
    alpha_inversion_q = np.array(eigs)
    entropy_q = np.array(entropy_q)
    sigma_p_q = np.array(sigma_p_q)
    sigma_m_q = np.array(sigma_m_q)

    def an_e_2(g,A,t):
        gamma = laguerre(0)(g**2 * t**2)*np.exp(-(g**2 * t**2)/2)
        abs_g = np.abs(gamma)
        eigs_1 = 0.5*(1+abs_g)
        eigs_2 = 0.5*(1-abs_g)
        
        svn = -eigs_1*np.log(eigs_1) - eigs_2*np.log(eigs_2)
        
        return svn
    
    
    
    colors = ['midnightblue']
    legend_handles = []
    
    fig, axs = plt.subplots()
    
    g_val = gvals_log[0]
    entropy_analytical = an_e_2(g_val,A_vals[0],time)
    #inversion_an = analytical_inversion(g_val,A_vals[0],time)
    
    entropy_ft = fft(np.abs(eigs))
    frequencies = np.fft.fftfreq(len(time), d=(time[1] - time[0]))
    
    correlation = correlate(entropy_q[0],entropy_q[0],mode='full')
    lags = np.arange(-len(entropy_q[0]) + 1, len(entropy_q[0]))
    
    
        
        
    axs.plot(time,entropy_q[0],color='darkgreen')
    #axs.plot(lags[5000:],correlation[5000:],color='darkgreen')
    #axs.plot(frequencies,entropy_ft,color='darkgreen')
    #axs.set_xlim(0,3.14)
    #axs.plot(time,atomic_inversion_q[0])
    #axs.plot(time,eigs)
    #axs.plot(time,shann,color='magenta')
    #axs.plot(time,shann_sc,color='midnightblue')
    #axs.plot(time,entropy_analytical,color='magenta',label='Predicted S')
    #axs.grid(axis='y')
    #axs.set_xlabel('time',fontsize=14)
    #axs.set_ylabel('S',fontsize=14)
    #axs.legend(fontsize=14)
    
    
    

    
    
    for g in range(len(gvals_log)):
        
        g_val = gvals_log[g]
        entropy_analytical = an_e_2(g_val,A_vals[0],time)
        #inversion_an = analytical_inversion(g_val,A_vals[0],time)
        #print(np.mean(entropy_analytical[1:-1])-np.mean(entropy_q[g])/np.mean(entropy_q[g]))
        '''
        axs[g,0].plot(time,entropy_q[g],label=f'g={g_val}',color='midnightblue')
        axs[g,0].plot(time,entropy_analytical,color='red',ls='--')
        axs[g,0].set_xlabel('t')
        axs[g,0].set_ylabel('S')
        
        axs[g,1].plot(time,atomic_inversion_q[g],color='midnightblue',ls='--')
        #axs[g,1].plot(time,analytical_sigmap(g_val,A_vals[0],time),color='magenta',ls='--')
        axs[g,1].set_xlabel('t')
        axs[g,1].set_ylabel(r'$\langle \sigma_z \rangle$')
        
        #axs[g,2].plot(time,4*sigma_p_q[g]*sigma_m_q[g],color='midnightblue')
        axs[g,2].plot(time,((4*sigma_p_q[g]*sigma_m_q[g]+(atomic_inversion_q[g])**2)-analytical_inversion(g_val,A_vals[0],time)**2)+analytical_sigmapm(g_val,A_vals[0],time)/np.exp(-g_val**2 * time **2),color='magenta',ls='--')
        #axs[g,2].plot(time,(analytical_inversion(g_val,A_vals[0],time)**2)+analytical_sigmapm(g_val,A_vals[0],time),color='midnightblue',ls='--')
        '''
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        '''
        axs[g,1].plot(time,entropy_q[g],color='red',ls='--')
        axs[g,1].set_xlabel('time')
        axs[g,1].set_ylabel('Atomic Inversion')
        axs[g,1].set_title(f'{gvals_log[g]}')
        '''
        
        
        
        
    
        

    plt.tight_layout()
    #plt.savefig('Inversion_vs_t_10_period.pdf', format='pdf', dpi=400)
if __name__ == '__main__':
    main()