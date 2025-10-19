# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:15:09 2020

@author: Matheus Wenceslau and Caio Moura

"""
from numba import jit,prange
import numpy as np 

#Lembretes: Ordem de conectivadade hexa do Abaqus coincide com livro do Paulo (anti-horaria)
#Lembretes: Ordem de conectivadade hexa do Salaome não coincide com livro do Paulo (horaria) 


#-----------------------------------------------------------------------------

@jit(nopython=True,cache=True)
def KGlobal(n_elem,K_e,Ke_all_elem,DOF_elem,order_I_COO_matrix,order_J_COO_matrix,
            K_glob_COO,K_glob_I,K_glob_J,ke_vector,exist_BC_Dirichlet_X,
            Dirichlet_elements_and_n_DOF_shared,Ke_with_Dirichlet,Ke_vector_Dirichlet,
            K_glob_COO_with_Dirichlet):
    """
    Function to assembly global stifiness matrix.  
    
    """
    
    n_values_matrix_elem=DOF_elem*DOF_elem
    
    # ke_all_elements[:,:]=0
    kk=0
    for M in range(n_elem):
        K_e[:,:]=Ke_all_elem[M*DOF_elem:(M*DOF_elem+DOF_elem),:]
                                                
        #COO MATRIX    
        ke_vector[:]=K_e[:,:].flatten()
        IE=order_I_COO_matrix[M,:]
        JE=order_J_COO_matrix[M,:]
        K_glob_COO[kk:(kk+n_values_matrix_elem)]=ke_vector[:]
        K_glob_I[kk:(kk+n_values_matrix_elem)]=IE[:]
        K_glob_J[kk:(kk+n_values_matrix_elem)]=JE[:]
                        
        if exist_BC_Dirichlet_X:   
            n_matrix=1
            if M in Dirichlet_elements_and_n_DOF_shared[:,0]: 
                # Ke_with_Dirichlet[:,:]=ke_all_elements[init:last,:]
                Ke_with_Dirichlet[:,:]=K_e[:,:]               
                elem_where=np.where(Dirichlet_elements_and_n_DOF_shared[:,0]==M)[0][0]
                for nn in range(Dirichlet_elements_and_n_DOF_shared.shape[1]-1):
                    n_nodes_shared=Dirichlet_elements_and_n_DOF_shared[elem_where,nn+1]
                    if n_nodes_shared!=0:
                        Ke_with_Dirichlet[nn,:]=0
                        Ke_with_Dirichlet[:,nn]=0
                        Ke_with_Dirichlet[nn,nn]=1/(n_nodes_shared*n_matrix)
                Ke_vector_Dirichlet[:]=Ke_with_Dirichlet.flatten()
                K_glob_COO_with_Dirichlet[kk:(kk+n_values_matrix_elem)]=Ke_vector_Dirichlet[:]                  
            else:
                K_glob_COO_with_Dirichlet[kk:(kk+n_values_matrix_elem)]=ke_vector[:] 
                
        kk=kk+n_values_matrix_elem 
        
    return K_glob_COO,K_glob_I,K_glob_J,K_glob_COO_with_Dirichlet     
   
#-----------------------------------------------------------------------------
 
@jit(nopython=True,cache=True)
def get_Ke_all_and_B_all(gauss_coord,gauss_weight,elem_coord,jacobian,det_Jacobian,
                         deri_phi_param,deri_phi_real,B_elem,B_t,B_Gauss,K_e,
                         connectivity,nodes,n_nodes_elem,mesh_type,Ke_all_elem,
                         B_all_elem,n_elem,DOF_elem,DOF_node_elem,n_Gauss_elem,
                         DOF_stress_strain,tang_modu,
                         B_and_Ke_elem,mat_prop,get_Be,get_Ke,det_Jacobian_all,
                         thickness_vector=None):
     """
     Function compute Ke_all and B_all of all elements
        
     """
         
     n_compnts_B=DOF_stress_strain*n_Gauss_elem
             
     for M in range(n_elem):        
         #2D Analysis
         if DOF_node_elem==2:
               thickness=thickness_vector[M]
               # K_e,B_elem=B_and_Ke_elem(gauss_coord,
               #                      gauss_weight,elem_coord,connectivity[M,:],
               #                      jacobian,det_Jacobian,deri_phi_param,deri_phi_real,
               #                      B_elem,B_t,B_Gauss,nodes,tang_modu,mesh_type,K_e,thickness)
               
               B_elem,det_Jacobian=get_Be(gauss_coord,gauss_weight,elem_coord,connectivity[M,:],
                          jacobian,det_Jacobian,deri_phi_param,deri_phi_real,B_elem,
                          nodes,mesh_type,thickness)
               
               #Criar uma chamada aqui para chamar a rotina que calcula o modulo tangente 
               #de todos os pontos de Gauss do elemento em analise 
               
               K_e=get_Ke(gauss_weight,det_Jacobian,B_elem,B_t,B_Gauss,tang_modu,K_e,thickness)
               

         #3D Analysis 
         if DOF_node_elem==3:
             mesh_type=mesh_type
             thickness=thickness_vector[M]
             
             # K_e,B_elem=B_and_Ke_elem(gauss_coord,gauss_weight,
             #                         elem_coord,connectivity[M,:],
             #           jacobian,det_Jacobian,deri_phi_param,deri_phi_real,B_elem,
             #                       B_t,B_Gauss,nodes,tang_modu,mesh_type,K_e,thickness)
             
             B_elem,det_Jacobian=get_Be(gauss_coord,gauss_weight,elem_coord,connectivity[M,:],
                        jacobian,det_Jacobian,deri_phi_param,deri_phi_real,B_elem,
                        nodes,mesh_type,thickness)
             
               #Criar uma chamada aqui para chamar a rotina que calcula o modulo tangente 
               #de todos os pontos de Gauss do elemento em analise 
             
             K_e=get_Ke(gauss_weight,det_Jacobian,B_elem,B_t,B_Gauss,tang_modu,K_e,thickness)
         
         B_all_elem[M*n_compnts_B:(M*n_compnts_B+n_compnts_B),:]=B_elem[:,:]
         Ke_all_elem[M*DOF_elem:(M*DOF_elem+DOF_elem),:]=K_e[:,:]
         det_Jacobian_all[M,:]=det_Jacobian
     

     return Ke_all_elem,B_all_elem,det_Jacobian_all
#-----------------------------------------------------------------------------

@jit(nopython=True,cache=True)
def get_Internal_force_all(gauss_weight,det_Jacobian_all,B_all_elem,B_Gauss,B_t,Fi_elem,
                            stress_gauss_all,gauss_coord,elem_coord,jacobian,
                            mesh_type,DOF_node_elem,n_elem,
                            Internal_force,Fi_elem_all,stress_gauss_elem,n_Gauss_elem,
                            DOF_stress_strain,F_int_Glob,all_dof_maps,
                            thickness_vector=None):
    
      n_compnts_B=DOF_stress_strain*n_Gauss_elem
      n_Stress=DOF_stress_strain
      F_int_Glob[:]=0
      for M in range(n_elem):        
            thickness=thickness_vector[M]
            det_Jacobian=det_Jacobian_all[M,:]
            B_elem=B_all_elem[M*n_compnts_B:(M*n_compnts_B+n_compnts_B),:]

            for gauss in range(n_Gauss_elem):
                stress_gauss_elem[:,gauss]=stress_gauss_all[M,gauss*n_Stress:(gauss*n_Stress+n_Stress)]
               
            Fi_elem=Internal_force(gauss_weight,det_Jacobian,B_elem,B_Gauss,B_t,Fi_elem,
                                stress_gauss_elem,gauss_coord,elem_coord,
                                jacobian,mesh_type,thickness)
            
            Fi_elem_all[:,M]=Fi_elem[:,0]
            dof_element=all_dof_maps[M,:]
            F_int_Glob[dof_element,0]+=Fi_elem[:,0]

      return Fi_elem_all,F_int_Glob           
               


              
