# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:56:49 2020

@author: Matheus Wenceslau and Caio Moura
"""
import numpy as np 
from mesh import MeshFEM 
import linear_elasticity_iso_3D 
from solvers import static_nonlinear
import time


#Continuar arrumando removendo as inicializações de dentro das rotinas, e colocando apenas fora. 

start=time.time()

#Config Mesh
config_mesh={}
config_mesh['mesh_file_name']='plate2_with_hole_3D_hexa_100x50x6.inp' 
config_mesh['BC_Neumann_point_X_']=np.array([1500])
config_mesh['BC_Dirichlet_X_']=np.array([0])
config_mesh['BC_Dirichlet_Y_']=np.array([0])
config_mesh['BC_Dirichlet_Z_']=np.array([0])

config_mesh['analysis_dimension']='3D'
config_mesh['Thickness_Group_']=np.array([6])
mesh=MeshFEM(config_mesh)

#Config Material model  
# mat_prop[0] --> elastic modulus - E
# mat_prop[1] --> poisson - nu

material_model=linear_elasticity_iso_3D
mat_prop=np.array([210E3,0.29])

#Out file name 
out_file_name='FEM_out'

# --- CONFIGURAÇÃO DO SOLUCIONADOR ---
# Altere esta variável para escolher o método de solução:
# 'direct':    Usa o solucionador esparso direto (spsolve), como no código original.
# 'iterative': Usa um solucionador iterativo (CG com precondicionador Jacobi), ideal para sistemas grandes.
SOLVER_METHOD = 'iterative' 
# ------------------------------------

#Solver and Poss-processing

result=static_nonlinear(mesh,material_model,mat_prop,out_file_name,method=SOLVER_METHOD)


# end=time.time()-start
