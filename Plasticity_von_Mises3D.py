# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16  2025

Matheus Wenceslau and Caio Moura

Von mises Plasticity model with isotropic and kinematic hardening

"""
import numpy as np
from numba import jit
from numpy import linalg
import math 

#@jit(nopython=True,cache=True)
def tg_modulus(tangent_modulus,mat_prop,internal_var_Gauss,strain_1_gauss,strain_0_gauss):
    
    Pro_DEV=np.zeros((6,6))
    Pro_DEV[0,0]=2/3
    Pro_DEV[0,1]=-1/3
    Pro_DEV[0,2]=-1/3
    Pro_DEV[1,0]=-1/3 
    Pro_DEV[1,1]=2/3
    Pro_DEV[1,2]=-1/3 
    Pro_DEV[2,0]=-1/3 
    Pro_DEV[2,1]=-1/3 
    Pro_DEV[2,2]=2/3
    Pro_DEV[3,3]=1/2
    Pro_DEV[4,4]=1/2
    Pro_DEV[5,5]=1/2
    
    I_voigt=np.zeros((6,1))
    I_voigt[0,0]=1
    I_voigt[1,0]=1
    I_voigt[2,0]=1

    E=mat_prop[0]
    Poisson = mat_prop[1]
    Sigma_Y0 = mat_prop[2]
    H = mat_prop[3]
    K_1 = mat_prop[4]
    K_2 = mat_prop[5]
    theta = mat_prop[6]
    
    G=E/(2*(1+Poisson))
    K_vol=2*G*(1+Poisson)/(3*(1-2*Poisson))
    
    #1. Recovering internal variables at Gauss point 
        
    #Elastic strain last strain
    elastic_strain_0=internal_var_Gauss[0:6]
    
    #Kinematic hardening tensor internal variable in the last time step
    Kin_harden_beta_0=internal_var_Gauss[6:12]
    
    #Isotropic hardening scalar internal variable in the last time step
    iso_harden_alfa_0=internal_var_Gauss[12]
    
    #Plastic multiplier
    delta_gama_0=internal_var_Gauss[13]
    
    #2.  Strain increment
    delta_strain=strain_1_gauss-strain_0_gauss
    
    #Converting vector in Voigt to tensor notation 
    
    Kin_harden_beta_0_tensor=vec_to_matrix(Kin_harden_beta_0,'Stress')
    
    elastic_strain_0_tensor=vec_to_matrix(elastic_strain_0,'Strain')
    
    delta_strain_tensor=vec_to_matrix(delta_strain,'Strain')
    
    #3.  Computing trial state
    
    #Elastic Trial strain
    strain_elas_trial=elastic_strain_0_tensor+delta_strain_tensor
    
    #Trace of the strain tensor 
    trace_e=np.trace(strain_elas_trial)
    
    #Trial Volumetric stress 
    S_vol_elas_trial=K_vol*trace_e*np.eye(3)
    
    #Trial deviatoric strain tensor
    ed_elas_trial=strain_elas_trial-1/3*trace_e*np.eye(3)
    
    #Trial deviatoric stress tensor 
    S_trial_dev=2*G*ed_elas_trial
    
    beta_trial=Kin_harden_beta_0_tensor
    
    #Trial flux Matrix
    eta_trial=S_trial_dev-beta_trial
    
    #L2-Norm of eta flux Matrix
    norm_eta_trial=linalg.norm(eta_trial)
    normalized_eta_trial=eta_trial/norm_eta_trial
    root_3_2=np.sqrt(3/2)
    
    #4.  Check plastic admissibility
    [fun_escoamento_trial,deri_sigma_alfa]=fun_encruamento_exp(iso_harden_alfa_0,Sigma_Y0,K_1,K_2,theta)
       
    #q Trial
    q_trial=root_3_2*norm_eta_trial
       
    #Yield surface trial
    f_trial=q_trial - fun_escoamento_trial 
    
    a=2*G*(1-delta_gama_0*3*G/q_trial)
    b=6*G**2*(delta_gama_0/q_trial-1/(3*G+H+deri_sigma_alfa))
    c=K_vol
    
    tangent_modulus=a*Pro_DEV + b*(normalized_eta_trial @ normalized_eta_trial.T) + c*(I_voigt @ I_voigt.T) #Verificar melhor se a normal N é somente isso mesmo 
       
    return tangent_modulus

#@jit(nopython=True,cache=True)    
def get_stress_and_strain(B_all_elem,tangent_modulus,stress_1,strain_1,connectivity,
                          n_nodes_element,n_gauss,DOF,u_glob,u_elem,n_elem,mat_prop,
                          internal_var_Gauss_1,internal_Var_1_all):
    
    """ 
    Function to compute stress and strain at gauss points 
    """
    
    n_Var_inter=13
    
    for elem in range(n_elem):
        u_elem=get_u_elem(connectivity[elem,:],n_nodes_element,DOF,u_glob,u_elem)
        u_elem_contig=np.ascontiguousarray(u_elem)
        B_elem=B_all_elem[elem*n_gauss*6:(elem*n_gauss*6+n_gauss*6),:]
      
        for gauss in range(n_gauss):
          
            #B_ele[0:6,:]= B matrix of the first guass point
            B_Gauss=np.ascontiguousarray(B_elem[gauss*6:(gauss*6+6),:])
            
            #Strain for the last time step
            strain_0_gauss=strain_1[elem,gauss*6:(gauss*6+6)]
            
            #Internal variable for the last time step
            internal_var_Gauss_0=internal_Var_1_all[elem,gauss*n_Var_inter:(gauss*n_Var_inter+n_Var_inter)]
            
            #Strain for the current time step
            strain_1_gauss=B_Gauss @ u_elem_contig
            
            #Call of the constitutive model
            stress_1_gauss,internal_var_Gauss_1=Plasticity_von_Mises3D(strain_1_gauss,
                            strain_0_gauss,internal_var_Gauss_0,mat_prop)
            
            #Update
            strain_1[elem,gauss*6:(gauss*6+6)]=strain_1_gauss[:]
            stress_1[elem,gauss*6:(gauss*6+6)]=stress_1_gauss[:]
            internal_Var_1_all[elem,gauss*n_Var_inter:(gauss*n_Var_inter+n_Var_inter)]=internal_var_Gauss_1[:]  
                                           

#@jit(nopython=True,cache=True)
def get_u_elem(connectivity,n_nodes_element,DOF,u_glob,u_elem):
    
    """
    Function to get element displacement
    """
    cont=0
    for i in range(n_nodes_element):
        for j in range(DOF):
            GDL=connectivity[i]*DOF+j
            u_elem[cont]=u_glob[GDL]
            cont=cont+1
    return u_elem

#@jit(nopython=True,cache=True)
def Plasticity_von_Mises3D(strain_1_gauss,strain_0_gauss,internal_var_Gauss,mat_prop):
    
    """
    strain_1_gauss: Updated Total strain at gauss point (updated);
    stress_1_gauss: Updated Stress at gauss point (updated); 
    strain_0_gauss: Total strain in last time step at gauss point;
    elastic_strain_0: Elastic strain in last time step at gauss point;
    Kin_harden_beta_0: Kinematic hardening tensor in last time step at gauss point;
    iso_harden_alfa_0: Isotropic hardening scalar in last time step at gauss point;
    
    """
    strain_1_gauss=np.array([-0.002500000000000,-0.002500000000000,0.010666666666667,1.761828530288945e-19,2.168404344971009e-19,-3.252606517456513e-19])
    strain_0_gauss=np.array([-0.002500000000000,-0.002500000000000,0.010000000000000,1.761828530288945e-19,2.168404344971009e-19,-3.252606517456513e-19])
    E=mat_prop[0]
    Poisson = mat_prop[1]
    Sigma_Y0 = mat_prop[2]
    H = mat_prop[3]
    K_1 = mat_prop[4]
    K_2 = mat_prop[5]
    theta = mat_prop[6]
    
    G=E/(2*(1+Poisson))
    K_vol=2*G*(1+Poisson)/(3*(1-2*Poisson))
    K_vol_2=E/(3*(1-2*Poisson))
    
    #1. Recovering internal variables at Gauss point 
       
    #Elastic strain last strain
    elastic_strain_0=internal_var_Gauss[0:6]
    elastic_strain_0=np.array([-0.002500000000000,-0.002500000000000,0.010000000000000,1.761828530288945e-19,2.168404344971009e-19,-3.252606517456513e-19])
    
    #Kinematic hardening tensor internal variable in the last time step
    Kin_harden_beta_0=internal_var_Gauss[6:12]
    
    #Isotropic hardening scalar internal variable in the last time step
    iso_harden_alfa_0=internal_var_Gauss[12]
    
    #2.  Strain increment
    delta_strain=strain_1_gauss-strain_0_gauss
    
    #Converting vector in Voigt to tensor notation 
    
    Kin_harden_beta_0_tensor=vec_to_matrix(Kin_harden_beta_0,'Stress')
    
    elastic_strain_0_tensor=vec_to_matrix(elastic_strain_0,'Strain')
    
    delta_strain_tensor=vec_to_matrix(delta_strain,'Strain')
    
    #3.  Computing trial state
    
    #Elastic Trial strain
    strain_elas_trial=elastic_strain_0_tensor+delta_strain_tensor
    
    #Trace of the strain tensor 
    trace_e=np.trace(strain_elas_trial)
    
    #Trial Volumetric stress 
    S_vol_elas_trial=K_vol*trace_e*np.eye(3)
    
    #Trial deviatoric strain tensor
    ed_elas_trial=strain_elas_trial-1/3*trace_e*np.eye(3)
    
    #Trial deviatoric stress tensor 
    S_trial_dev=2*G*ed_elas_trial
    
    beta_trial=Kin_harden_beta_0_tensor
    alfa_trial=iso_harden_alfa_0
    
    #Trial flux Matrix
    eta_trial=S_trial_dev-beta_trial
    
    #L2-Norm of eta flux Matrix
    norm_eta_trial=linalg.norm(eta_trial)
    normalized_eta_trial=eta_trial/norm_eta_trial
    root_3_2=np.sqrt(3/2)
    
    Normal=root_3_2*normalized_eta_trial
    
    #q Trial
    q_trial=root_3_2*norm_eta_trial
    
    #4.  Check plastic admissibility
    
    #Yield function
    fun_escoamento_trial,_=fun_encruamento_exp(alfa_trial,Sigma_Y0,K_1,K_2,theta)
    
    #Yield surface trial
    f_trial=q_trial - fun_escoamento_trial 
    
    if f_trial <=0:
        #Elastic
        sigma=S_trial_dev+S_vol_elas_trial
        ee_1=strain_elas_trial
        alfa_1=iso_harden_alfa_0
        beta_1=Kin_harden_beta_0_tensor
        delta_gama=0
    else:
        #Plastic - Corrector phase            
        delta_gama=newton_delta_gama(q_trial,alfa_trial,Sigma_Y0,K_1,K_2,theta,G,H)
     
        alfa_1=iso_harden_alfa_0+delta_gama
        beta_1=Kin_harden_beta_0_tensor+2/3*H*delta_gama*Normal
        S_dev=S_trial_dev-2*G*delta_gama*Normal
        sigma=S_dev+S_vol_elas_trial 
        # ep_1=plastic_strain_0+delta_gama*Normal
        ee_1=1/(2*G)*S_dev+1/3*trace_e*np.eye(3)
        # ee_1_=strain_1_gauss-matrix_to_vec(ep_1,'Strain')
            
    #Returning to Voigt notation 
    stress_Gauss_1=matrix_to_vec(sigma,'Stress')
    # internal_var_Gauss_1[0:6]=matrix_to_vec(ep_1,'Strain')
    internal_var_Gauss[0:6]=matrix_to_vec(ee_1,'Strain')
    internal_var_Gauss[6:12]=matrix_to_vec(beta_1,'Stress')
    internal_var_Gauss[12]=alfa_1
    internal_var_Gauss[13]=delta_gama
        
    return sigma,internal_var_Gauss
    
    
#--------------------- Função de encruamento exponencial --------------------#     

#@jit(nopython=True,cache=True)
def fun_encruamento_exp(alfa,Sigma_Y0,K_1,K_2,theta):
    """
    Metodo função de encruamento exponencial e para obter derivada da função de 
    encruamento em relação a alfa (parametro de encruamento isotropico)
   
    """
    Sigma_Y=Sigma_Y0+K_1*alfa+K_2*(1-math.exp(-theta*alfa))  
    
    deri_sigma_alfa=(K_1+theta*K_2*math.exp(-alfa*theta))
   
    return Sigma_Y,deri_sigma_alfa    

# ----------- Newton para obter encruamento isotropico -----------------------#

#@jit(nopython=True,cache=True)
def newton_delta_gama(q_trial,alfa_0,Sigma_Y0,K_1,K_2,theta,G,H):
    
    delta_gama=0
    alfa_1=alfa_0
    N_iter=100
    for k in range(N_iter):
        
        [fun_escoamento,deri_sigma_alfa]=fun_encruamento_exp(alfa_1,Sigma_Y0,K_1,K_2,theta)
       
        #Criterio de escoamento
        f_k1=q_trial-delta_gama*(3*G+H)-fun_escoamento 
        
        #Verificacao se encontrou delta_gama raiz da equacao 
        
        if linalg.norm(f_k1)<1E-6:
            break 
         
        derivada_f_k1=-(3*G+H)-deri_sigma_alfa
        incr_delta_gama=-f_k1/derivada_f_k1
        delta_gama=delta_gama+incr_delta_gama  
        alfa_1=alfa_0+delta_gama
         
    if linalg.norm(f_k1)<1E-6:
        fail=False
    else:
        print(['Newton modelo Plastico saiu com residuo maior que 1E-6 '])
        fail=True
        
    return delta_gama

#----------------------------------------------------------------------------#    

#@jit(nopython=True,cache=True)    
def vec_to_matrix(v,flag):
    #v=[v_11 v_22 v_33 v_23 v_13 v_12]
    
    M=np.zeros((3,3))
    
    if flag=="Strain":
        factor=2
    else: 
        factor=1

    for i in range(3):
        M[i,i]=v[i]
     
    M[0,1]=v[5]/factor
    M[1,2]=v[3]/factor
    M[0,2]=v[4]/factor
    
    M[1,0]=M[0,1]
    M[2,1]=M[1,2]
    M[2,0]=M[0,2]
   
    return M    

#----------------------------------------------------------------------------#  

#@jit(nopython=True,cache=True)   
def matrix_to_vec(M,flag): 
    #v=[v_11 v_22 v_33 v_23 v_13 v_12]
    
    v=np.zeros((6))   
    
    if flag=="Strain":
        factor=2
    else: 
        factor=1
        
    
    for i in range(3):
        v[i]=M[i,i]
                
    v[3]=M[1,2]*factor
    v[4]=M[0,2]*factor 
    v[5]=M[0,1]*factor         
    
    return v
    
    
    
    