# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:25:58 2020

@author: Matheus Wenceslau and Caio Moura
"""
from init_Vars import Init_Vars
from KGlob_Assembly_JIT_Numba import KGlobal,get_Ke_all_and_B_all,get_Internal_force_all
from scipy.sparse import coo_matrix,linalg
from inload import Inload 
import pos_processing
import numpy as np 
import time 
from scipy.sparse.linalg import gmres, spilu, LinearOperator
from scipy.sparse.linalg import cg, LinearOperator 

"""
Implemented solvers: 
    Static linear - Infinitesimal deformation: static_linear 
To do:
    Static non linear  - Infinitesimal deformation
    Explicit linear dynamics  - Infinitesimal deformation
    Implicit linear dynamics  - Infinitesimal deformation
"""

def static_linear(mesh,material_model,mat_prop,out_file_name,method='direct'):
        
    #External force
    external_force=Inload(mesh)
    load_vector=external_force.load_vector
    
    #Global Arrays
    Vars=Init_Vars(mesh,material_model)
        
    #Elementary stifiness (Ke) and derivative matrix (B)
    element=mesh.fun_elem
    
    connectivity=mesh.connectivity
    nodes=mesh.nodes
    n_nodes_elem=mesh.n_nodes_elem
    mesh_type=mesh.mesh_type
    Ke_all_elem=Vars.Ke_all_elem
    B_all_elem=Vars.B_all_elem
    n_elem=mesh.n_elem
    DOF_elem=mesh.DOF_elem
    DOF_node_elem=mesh.DOF_node_elem
    n_Gauss_elem=mesh.n_Gauss_elem
    DOF_stress_strain=mesh.DOF_stress_strain
    tang_modu=material_model.tg_modulus(Vars.tang_modu,mat_prop) 
    B_and_Ke_elem=element.B_and_Ke_elem
    get_Be=element.get_Be
    get_Ke=element.get_Ke
    Internal_force=element.Internal_force
    
    #2D Analysis
    if mesh.DOF_node_elem==2:        
        thickness_vector=mesh.thickness_vector        
    #3D Analysis
    elif  mesh.DOF_node_elem==3:        
        thickness_vector=np.ones(mesh.n_elem) #created only to avoid vector search errors.         
               
    Ke_all_elem,B_all_elem,det_Jacobian_all=get_Ke_all_and_B_all(Vars.gauss_coord,
                                Vars.gauss_weight,Vars.elem_coord,
                                Vars.jacobian,Vars.det_Jacobian,Vars.deri_phi_param,
                                Vars.deri_phi_real,
                                Vars.B_elem,Vars.B_t,Vars.B_Gauss,Vars.K_e,
                                connectivity,nodes,n_nodes_elem,
                                mesh_type,Ke_all_elem,B_all_elem,n_elem,
                                DOF_elem,DOF_node_elem,n_Gauss_elem,
                                DOF_stress_strain,tang_modu,B_and_Ke_elem,
                                mat_prop,get_Be,get_Ke,Vars.det_Jacobian_all,
                                thickness_vector)
            
    order_I_COO_matrix=mesh.order_I_COO_matrix
    order_J_COO_matrix=mesh.order_J_COO_matrix
        
    #Global Sparse Matrix
    exist_BC_Dirichlet_X=True
    (K_glob_COO,K_glob_I,K_glob_J,K_glob_COO_with_Dirichlet)=KGlobal(mesh.n_elem,Vars.K_e,
                      Vars.Ke_all_elem,mesh.DOF_elem,order_I_COO_matrix,
                      order_J_COO_matrix,Vars.K_glob_COO,Vars.K_glob_I,
                      Vars.K_glob_J,Vars.ke_vector,exist_BC_Dirichlet_X,
                      mesh.Dirichlet_elements_and_n_DOF_shared,
                      Vars.Ke_with_Dirichlet,Vars.Ke_vector_Dirichlet,
                      Vars.K_glob_COO_with_Dirichlet) 
           
    K_glob_CSR=coo_matrix((K_glob_COO,(K_glob_I,K_glob_J)),shape=(mesh.DOF_tot,mesh.DOF_tot)).tocsr()
    K_glob_CSR_BC=coo_matrix((K_glob_COO_with_Dirichlet,(K_glob_I,K_glob_J)),shape=(mesh.DOF_tot,mesh.DOF_tot)).tocsr()
       
    #Imposing BC in load vector 
    load_subtraction=imposing_force_BC(mesh,K_glob_CSR,K_glob_CSR_BC,
                    external_force.load_vector,Vars.load_subtraction,Vars.cont,
                    mesh.Dirichlet_DOF)
  
    #Displacement     
    displacement=linear_system_solver_call(method,K_glob_CSR_BC,load_subtraction)
        
    #Pos-processing 
    result={}
    result['displacement']=displacement
    tang_modu=material_model.tg_modulus(Vars.tang_modu,mat_prop)
  
    stress_gauss_all,strain_gauss_all,stress_nodes,strain_nodes=pos_processing.pos_static_linear(
                 Vars.B_all_elem,tang_modu,Vars.stress_gauss_all,Vars.strain_gauss_all,
                 mesh.connectivity,mesh.n_nodes_elem,
                 mesh.n_Gauss_elem,mesh.DOF_node_elem,displacement,Vars.u_elem,
                 mesh.n_elem,material_model,Vars.stress_nodes,Vars.strain_nodes,
                 Vars.cont_average,Vars.extrapol_vec_stress,
                 Vars.extrapol_vec_strain,mesh.DOF_stress_strain,element,
                 Vars.N,Vars.phi_vec,Vars.gauss_coord,mesh.mesh_type)
    
    Vars.Fi_elem_all,Vars.F_int_Glob=get_Internal_force_all(Vars.gauss_weight,
                                det_Jacobian_all,B_all_elem,
                                Vars.B_Gauss,Vars.B_t,Vars.Fi_elem,
                                Vars.stress_gauss_all,Vars.gauss_coord,Vars.elem_coord,Vars.jacobian,
                                mesh_type,DOF_node_elem,n_elem,
                                Internal_force,Vars.Fi_elem_all,Vars.stress_gauss_elem,
                                mesh.n_Gauss_elem,mesh.DOF_stress_strain,
                                Vars.F_int_Glob,mesh.all_dof_maps,thickness_vector)
    #Residuo
    R=Vars.F_int_Glob[:,0]-load_vector
    R[mesh.Dirichlet_DOF]=0
    # # Check if all elements in 'arr' are close to zero
    all_close_to_zero = np.allclose(R, 0, rtol=1e-5, atol=1e-6)
    print(all_close_to_zero)  
    
    result['stress_gauss']=stress_gauss_all
    result['strain_gauss']=strain_gauss_all
    result['stress_nodes']=stress_nodes
    result['strain_nodes']=strain_nodes
    
 
    #Saving results 
    pos_processing.save_results(mesh,displacement,stress_nodes,strain_nodes,out_file_name)
            
    return result
        
#-----------------------------------------------------------------------------#   

def static_nonlinear(mesh, material_model, mat_prop, out_file_name, method='direct', max_iter=100, tolerance=1e-5):
    
    """
    Solucionador estático não linear usando o método de Newton-Raphson.
    """
    print("\n--- INICIANDO SOLUÇÃO ESTÁTICA NÃO LINEAR (NEWTON-RAPHSON) ---")
    
    # 1. INICIALIZAÇÃO
   
    # Variáveis globais e elementares
    Vars = Init_Vars(mesh, material_model)
    element = mesh.fun_elem
    
    # Deslocamento inicial
    displacement = np.zeros(mesh.DOF_tot)
       
    B_and_Ke_elem=element.B_and_Ke_elem
    get_Be=element.get_Be
    get_Ke=element.get_Ke
    Internal_force=element.Internal_force
       
    #2D Analysis
    if mesh.DOF_node_elem==2:        
        thickness_vector=mesh.thickness_vector        
    #3D Analysis
    elif  mesh.DOF_node_elem==3:        
        thickness_vector=np.ones(mesh.n_elem) #created only to avoid vector search errors.  
        
    # Forças externas (constantes durante a análise)
    external_force = Inload(mesh)
    F_ext = external_force.load_vector        
                       
    # 2. LOOP DE NEWTON-RAPHSON
    for i in range(max_iter):
        print(f"\n--- Iteração {i+1} ---")
        
        #Inserir multiplicação do vetor força externa por um fator de carga 
                
        # Zera forças internas para a nova iteração
        Vars.F_int_Glob[:,0] = 0
        
        # a. MONTAGEM DAS FORÇAS INTERNAS E MATRIZ TANGENTE
        # NOTA: Em um problema não linear real, a 'tang_modu' seria atualizada
        # dentro do loop do elemento com base no estado de deformação atual.
        #Esta chamada deve ser movida para dentro da função "get_Ke_all_and_B_all" 
        #ou dentro da Ke_matrix (nessa fica ate mais visual)
        # tang_modu = material_model.tg_modulus(Vars.tang_modu, mat_prop)
        
        tang_modu=Vars.tang_modu
        
        # Recalcula a rigidez elementar para todos os elementos
        #passar vetor de deslocamento do ultimo passo de newton
        #passar variaveis internas do ultimo passo de newton
        #passar delta gamma da ultima iteracao de newton. Se for a primeira do passo de carga, deve ser igual a zero 
        #chama dentro da rotina do elemento a rotina que calcula o modulo tangente
        Vars.Ke_all_elem,Vars.B_all_elem,Vars.det_Jacobian_all=get_Ke_all_and_B_all(Vars.gauss_coord,
                                    Vars.gauss_weight,Vars.elem_coord,
                                    Vars.jacobian,Vars.det_Jacobian,Vars.deri_phi_param,
                                    Vars.deri_phi_real,
                                    Vars.B_elem,Vars.B_t,Vars.B_Gauss,Vars.K_e,
                                    mesh.connectivity,mesh.nodes,mesh.n_nodes_elem,
                                    mesh.mesh_type,Vars.Ke_all_elem,Vars.B_all_elem,mesh.n_elem,
                                    mesh.DOF_elem,mesh.DOF_node_elem,mesh.n_Gauss_elem,
                                    mesh.DOF_stress_strain,tang_modu,B_and_Ke_elem,
                                    mat_prop,get_Be,get_Ke,Vars.det_Jacobian_all,
                                    thickness_vector)
        
        # b. MONTAGEM E SOLUÇÃO DO SISTEMA LINEAR
        # Monta a matriz de rigidez global K_t para a iteração
        exist_BC_Dirichlet_X=True
        (K_glob_COO, K_glob_I, K_glob_J, K_glob_COO_with_Dirichlet) = KGlobal(mesh.n_elem,
            Vars.K_e, Vars.Ke_all_elem, mesh.DOF_elem, mesh.order_I_COO_matrix, mesh.order_J_COO_matrix,
            Vars.K_glob_COO, Vars.K_glob_I, Vars.K_glob_J, Vars.ke_vector, exist_BC_Dirichlet_X,
            mesh.Dirichlet_elements_and_n_DOF_shared, Vars.Ke_with_Dirichlet,
            Vars.Ke_vector_Dirichlet, Vars.K_glob_COO_with_Dirichlet)

        K_glob_CSR=coo_matrix((K_glob_COO,(K_glob_I,K_glob_J)),shape=(mesh.DOF_tot,mesh.DOF_tot)).tocsr()
        K_glob_CSR_BC = coo_matrix((K_glob_COO_with_Dirichlet, (K_glob_I, K_glob_J)), shape=(mesh.DOF_tot, mesh.DOF_tot)).tocsr()   
        
        #Stress and strain in gauss points 
        stress_gauss_all,strain_gauss_all=material_model.get_stress_and_strain(Vars.B_all_elem,
                  tang_modu,Vars.stress_gauss_all,Vars.strain_gauss_all,mesh.connectivity,mesh.n_nodes_elem,
                          mesh.n_Gauss_elem,mesh.DOF_node_elem,displacement,Vars.u_elem,mesh.n_elem,
                          mat_prop,Vars.internal_var_Gauss_1,Vars.internal_Var_1_all)
        

        # Monta o vetor de força interna global
        Vars.Fi_elem_all,Vars.F_int_Glob=get_Internal_force_all(Vars.gauss_weight,Vars.det_Jacobian_all,Vars.B_all_elem,
                                Vars.B_Gauss,Vars.B_t,Vars.Fi_elem,
                                Vars.stress_gauss_all,Vars.gauss_coord,Vars.elem_coord,Vars.jacobian,
                                mesh.mesh_type,mesh.DOF_node_elem,mesh.n_elem,
                                Internal_force,Vars.Fi_elem_all,Vars.stress_gauss_elem,
                                mesh.n_Gauss_elem,mesh.DOF_stress_strain,Vars.F_int_Glob,mesh.all_dof_maps,
                                thickness_vector)
            
        # c. CÁLCULO DO RESÍDUO
        R = F_ext - Vars.F_int_Glob[:,0]
        
        # Impõe Condições de Contorno no vetor de resíduo
        R_with_Dirichlet=imposing_force_BC(mesh,K_glob_CSR,K_glob_CSR_BC,
                        R,Vars.load_subtraction,Vars.cont,mesh.Dirichlet_DOF)
            
        # d. VERIFICAÇÃO DE CONVERGÊNCIA
        norm_R = np.linalg.norm(R_with_Dirichlet)
        print(f"  Norma do Resíduo: {norm_R:.6e}")
        if norm_R < tolerance:
            print(f"\nSolução convergiu em {i} iterações.")
            break
                    
        # Resolve K_t * delta_u = R
        delta_u = linear_system_solver_call(method, K_glob_CSR_BC, R_with_Dirichlet)
        
        # e. ATUALIZAÇÃO DOS DESLOCAMENTOS
        displacement += delta_u

    else: # Se o loop terminar sem convergir
        print(f"\nAVISO: A solução NÃO convergiu após {max_iter} iterações.")

    # 3. PÓS-PROCESSAMENTO
    result = {}
    result['displacement'] = displacement
    tang_modu_final = material_model.tg_modulus(Vars.tang_modu, mat_prop)

    stress_gauss_all, strain_gauss_all, stress_nodes, strain_nodes = pos_processing.pos_static_linear(
        Vars.B_all_elem, tang_modu_final, Vars.stress_gauss_all, Vars.strain_gauss_all,
        mesh.connectivity, mesh.n_nodes_elem, mesh.n_Gauss_elem, mesh.DOF_node_elem,
        displacement, Vars.u_elem, mesh.n_elem, material_model, Vars.stress_nodes,
        Vars.strain_nodes, Vars.cont_average, Vars.extrapol_vec_stress,
        Vars.extrapol_vec_strain, mesh.DOF_stress_strain, element, Vars.N,
        Vars.phi_vec, Vars.gauss_coord, mesh.mesh_type)
    
    result['stress_gauss'] = stress_gauss_all
    result['strain_gauss'] = strain_gauss_all
    result['stress_nodes'] = stress_nodes
    result['strain_nodes'] = strain_nodes
 
    pos_processing.save_results(mesh, displacement, stress_nodes, strain_nodes, out_file_name)
            
    return result     

def imposing_force_BC(mesh,KGlob_csc,KGlob_csc_BC,load_vector,load_subtraction,cont,
                      Dirichlet_DOF):
                    
    """
    Function to impose Dirichlet BC on the load vector. 
       
    """
    if np.all(mesh.Dirichlet_values==0):
        load_subtraction=load_vector
        load_subtraction[Dirichlet_DOF]=0
    else: 
        cont[0]=0
        for i in mesh.Dirichlet_DOF:
            BC_value=mesh.Dirichlet_values[cont[0]]
            KGlob_csc[i,i]=0
            load_subtraction+=KGlob_csc[:,i].toarray().reshape(-1)*(-BC_value)
            load_subtraction[i]=0
            cont[0]+=1
        load_subtraction=load_vector+load_subtraction
        
        cont[0]=0
        for i in mesh.Dirichlet_DOF:
            BC_value=mesh.Dirichlet_values[cont[0]]
            load_subtraction[i]=BC_value*KGlob_csc_BC[i,i]
            cont[0]+=1
    return load_subtraction

#-----------------------------------------------------------------------------# 

def linear_system_solver_call(method,K_glob_CSR_BC,load_subtraction):

    if method == 'direct':
        print("\n--- Resolvendo com Método Direto (spsolve) ---")
        start_solve_time = time.time()
        
        displacement = linalg.spsolve(K_glob_CSR_BC,load_subtraction)
        
        end_solve_time = time.time()
        print(f"Solução direta encontrada em {end_solve_time - start_solve_time:.4f} segundos.")

    elif method == 'iterative':
        print("\n--- Resolvendo com Iterativo (CG + Precondicionador Jacobi) ---")
        start_solve_time = time.time()
        
        precon="Jacobi"
        
        if precon == 'Jacobi':
            # --- Precondicionador de Jacobi (cálculo instantâneo) ---
            # Extrai a diagonal da matriz K. Adiciona-se um valor pequeno (epsilon)
            # para evitar qualquer divisão por zero, embora seja raro em matrizes de rigidez.
            diag_K = K_glob_CSR_BC.diagonal()
            
            # O precondicionador (M) é um operador que resolve M*x=b. Para Jacobi,
            # isso é simplesmente uma divisão pela diagonal.
            M = LinearOperator(shape=K_glob_CSR_BC.shape, matvec=lambda r: r / diag_K)
            print("Precondicionador de Jacobi (diagonal) criado instantaneamente.")
            
        elif 'ILU':
            try:
                K_g_csc = K_glob_CSR_BC.tocsc()
                # Tente deixar o precondicionador um pouco mais robusto
                # Diminuir drop_tol ou aumentar fill_factor melhora a precisão do ILU
                print("Calculando o precondicionador ILU (pode levar um tempo)...")
                drop_tol_value=1e-3
                fill_factor_value=30
                ilu = spilu(K_g_csc, drop_tol=drop_tol_value, fill_factor=fill_factor_value) # Valores ajustados
                M = LinearOperator(shape=K_glob_CSR_BC.shape, matvec=ilu.solve)
                print("Precondicionador ILU calculado com sucesso.")
            except (RuntimeError, ValueError) as e:
                print(f"Aviso: Falha ao calcular o precondicionador ILU ({e}).")
                M = None            
        
        # --- Solucionador de Gradiente Conjugado (CG) ---
        # É o mais indicado para matrizes simétricas definidas-positivas.
        
        # Callback para monitorar o progresso a cada 20 iterações
        residuals = []
        def callback_res(xk):
            # O callback do CG recebe a solução atual (xk), não o resíduo.
            # Calculamos o resíduo para monitoramento.
            if len(residuals) % 20 == 0:
                res_norm = np.linalg.norm(load_subtraction - K_glob_CSR_BC.dot(xk))
                print(f"  Iteração CG {len(residuals)+1}, Norma do Resíduo: {res_norm:.4e}")
            residuals.append(0) # Apenas para contar as iterações

        print("Iniciando solucionador de Gradiente Conjugado (CG)...")
        displacement, exit_code = cg(K_glob_CSR_BC, load_subtraction, M=M, tol=1e-7, maxiter=5000, callback=callback_res)
        
        end_solve_time = time.time()
        
        if exit_code == 0:
            print(f"\nSolução iterativa convergiu em {len(residuals)} iterações.")
            print(f"Solução encontrada em {end_solve_time - start_solve_time:.4f} segundos.")
        else:
            print(f"\nERRO: A solução iterativa NÃO convergiu em {len(residuals)} iterações (código: {exit_code}).")

    else:
        raise ValueError(f"Método de solução '{method}' não reconhecido. Use 'direct' ou 'iterative'.")    
        
    return displacement
