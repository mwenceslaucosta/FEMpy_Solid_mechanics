# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:56:49 2020

@author: Matheus Wenceslau and Caio Moura
"""
import numpy as np 
from mesh import MeshFEM 
import plane_stress_lin_elast_iso_2D 
from solvers import static_linear

#Config Mesh
config_mesh={}
config_mesh['mesh_file_name']='plate2_with_hole_2D_quad_100x50x6.med' 
config_mesh['BC_Neumann_point_X_']=np.array([1500])
config_mesh['BC_Dirichlet_X_']=np.array([0])
config_mesh['BC_Dirichlet_Y_']=np.array([0])

config_mesh['analysis_dimension']='2D'
config_mesh['Thickness_Group_']=np.array([6])
mesh=MeshFEM(config_mesh)

#Config Material model  
# mat_prop[0] --> elastic modulus - E
# mat_prop[1] --> poisson - nu

material_model=plane_stress_lin_elast_iso_2D
mat_prop=np.array([210E3,0.29])

#Out file name 
out_file_name='FEM_out'

# --- CONFIGURAÇÃO DO SOLUCIONADOR ---
# Altere esta variável para escolher o método de solução:
# 'direct':    Usa o solucionador esparso direto (spsolve), como no código original.
# 'iterative': Usa um solucionador iterativo (GMRES com precondicionador ILU), ideal para sistemas grandes.
SOLVER_METHOD = 'direct' 
# ------------------------------------

#Solver and Poss-processing
result=static_linear(mesh,material_model,mat_prop,out_file_name,method=SOLVER_METHOD)


