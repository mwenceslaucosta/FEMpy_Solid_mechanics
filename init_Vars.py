# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:27:54 2020

@author: Matheus Wenceslau and Caio Moura
"""
import numpy as np
import sys 

#-----------------------------------------------------------------------------    

class Init_Vars:                                           
    def __init__(self,mesh,mat_model):
        
        #Initializing arrays used in the global stiffiness assembly
        n_positions=mesh.DOF_elem*mesh.DOF_elem*mesh.n_elem   
        self.coo_i=np.zeros(n_positions,dtype=np.int32)
        self.coo_j=np.zeros(n_positions,dtype=np.int32)
        self.coo_data=np.zeros(n_positions)
        self.coo_data_BC=np.zeros(n_positions)
        self.cont=np.zeros(3,dtype=np.int32)
        self.load_subtraction=np.zeros(mesh.DOF_tot)
        
        #Initializing arrays used in the elementary stiffiness 
        if mesh.DOF_node_elem==3:
            n_Voight=6
        elif mesh.DOF_node_elem==2:
            n_Voight=3
        else:
            sys.exit('Fatal error: Number of Voigth component do not match')  
        
        self.elem_coord=np.zeros((mesh.n_nodes_elem,mesh.DOF_node_elem))
        self.phi=np.zeros((mesh.n_nodes_elem,mesh.n_Gauss_elem))
        self.gauss_coord=np.zeros((mesh.n_Gauss_elem,mesh.DOF_node_elem))
        self.gauss_weight=np.zeros((mesh.n_Gauss_elem,mesh.DOF_node_elem))
        self.B_all_elem=np.zeros((n_Voight*mesh.n_Gauss_elem*mesh.n_elem,mesh.DOF_elem))
        self.Ke_all_elem=np.zeros((mesh.DOF_elem*mesh.n_elem,mesh.DOF_elem))
        self.K_e=np.zeros((mesh.DOF_elem,mesh.DOF_elem))
        self.N=np.zeros((mesh.n_nodes_elem,mesh.n_nodes_elem))
        self.phi_vec=N=np.zeros(mesh.n_nodes_elem)
        n_values_matrix_elem=mesh.DOF_elem*mesh.DOF_elem
        self.K_glob_COO = np.zeros((n_values_matrix_elem * mesh.n_elem), dtype=np.float64)              # Values of the global conductance matrix.
        self.K_glob_I = np.zeros((n_values_matrix_elem * mesh.n_elem), dtype=np.int32)                  # Row indices (I) of the global K matrix.
        self.K_glob_J = np.zeros((n_values_matrix_elem * mesh.n_elem), dtype=np.int32)                  # Column indices (J) of the global K matrix.
        self.ke_vector = np.zeros((mesh.DOF_elem * mesh.DOF_elem), dtype=np.float64)                    # Temporary vector for an element's stiffness matrix values.        
        self.Ke_with_Dirichlet = np.zeros((mesh.DOF_elem,mesh.DOF_elem), dtype=np.float64)
        self.Ke_vector_Dirichlet = np.zeros((mesh.DOF_elem * mesh.DOF_elem), dtype=np.float64)
        self.K_glob_COO_with_Dirichlet = np.zeros((n_values_matrix_elem * mesh.n_elem), dtype=np.float64)  
        self.Ke_all_elem_iter = np.zeros((mesh.DOF_elem*mesh.n_elem,mesh.DOF_elem)) # Matriz de rigidez da iteração
        self.det_Jacobian_all = np.zeros((mesh.n_elem,mesh.n_Gauss_elem))

        self.F_int_Glob=np.zeros((mesh.DOF_tot,1))    
        self.Fi_elem=np.zeros((mesh.DOF_elem,1))
        self.Fi_elem_all=np.zeros((mesh.DOF_elem,mesh.n_elem))
        self.jacobian=np.zeros((mesh.DOF_node_elem,mesh.DOF_node_elem))
        self.det_Jacobian=np.zeros(mesh.n_Gauss_elem)
        self.deri_phi_param=np.zeros((mesh.DOF_node_elem,mesh.n_nodes_elem))
        self.deri_phi_real=np.zeros((mesh.n_Gauss_elem*mesh.DOF_node_elem,mesh.n_nodes_elem))
        self.stress_gauss_elem=np.zeros((mesh.DOF_stress_strain,mesh.n_Gauss_elem))
        n_compnts_B=mesh.DOF_stress_strain*mesh.n_Gauss_elem

        self.B_elem=np.zeros((n_compnts_B,mesh.DOF_elem))
        self.B_t=np.zeros((mesh.DOF_elem,mesh.DOF_stress_strain))
        self.B_Gauss=np.zeros((mesh.DOF_stress_strain,mesh.DOF_elem))
        
        material_name=mat_model.__name__
        if material_name.endswith('3D') and mesh.DOF_node_elem==2:
            sys.exit('Fatal error: Constitutive model does not match mesh type')
        elif material_name.endswith('2D') and mesh.DOF_node_elem==3:
            sys.exit('Fatal error: Constitutive model does not match mesh type')
        
        if material_name == 'Plasticity_von_Mises3D':
            self.stress_gauss_all=np.zeros((mesh.n_elem,n_Voight*mesh.n_Gauss_elem))
            self.strain_gauss_all=np.zeros((mesh.n_elem,n_Voight*mesh.n_Gauss_elem))
            self.internal_Var_1_all=np.zeros((mesh.n_elem,13*mesh.n_Gauss_elem))
            self.tang_modu=np.zeros((6,6))
            self.internal_var_Gauss_1=np.zeros((13,1))
            self.stress_nodes=np.zeros((mesh.n_nodes_glob,n_Voight)) 
            self.strain_nodes=np.zeros((mesh.n_nodes_glob,n_Voight)) 
            self.cont_average=np.zeros(mesh.n_nodes_glob)
            self.extrapol_vec_strain=np.zeros(mesh.n_nodes_elem)
            self.extrapol_vec_stress=np.zeros(mesh.n_nodes_elem)
                       
        elif (material_name == 'linear_elasticity_iso_3D' or 
              material_name == 'plane_stress_lin_elast_iso_2D'):
            self.stress_gauss_all=np.zeros((mesh.n_elem,n_Voight*mesh.n_Gauss_elem))
            self.strain_gauss_all=np.zeros((mesh.n_elem,n_Voight*mesh.n_Gauss_elem))
            self.stress_nodes=np.zeros((mesh.n_nodes_glob,n_Voight)) 
            self.strain_nodes=np.zeros((mesh.n_nodes_glob,n_Voight)) 
            self.cont_average=np.zeros(mesh.n_nodes_glob)
            self.extrapol_vec_strain=np.zeros(mesh.n_nodes_elem)
            self.extrapol_vec_stress=np.zeros(mesh.n_nodes_elem)
            self.tang_modu=np.zeros((n_Voight,n_Voight))

        self.u_glob=np.zeros((n_Voight*mesh.n_Gauss_elem*2,mesh.n_elem))
        self.u_elem=np.zeros(mesh.DOF_node_elem*mesh.n_nodes_elem)
            
        
        
#-----------------------------------------------------------------------------
            

        
             
         
        
