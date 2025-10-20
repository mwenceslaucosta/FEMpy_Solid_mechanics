%*******************************************
% GRANTE
% Departamento de Engenharia Mec?nica
% UFSC-SC
%
% Prof. Eduardo Fancello
% ******************************************
% Programa Principal Elasto Plasticidade 3D
% ******************************************
clear;
clear global;
%close all;

% Dados
GetGlobal;
DadosCubo;

% --------------------------------------
% Gera matriz ID que contabiliza equacoes
% --------------------------------------
in = [1 2 3];
global ID;
ID = zeros(NumNos,3);
for n=1:NumNos
    ID(n,:)= in;
    in = in + 3;
end
NumGLTot = NumNos*3;

% ----------------------------------------------------
% Monta Vetor LM do elemento e coordenadas do elemento
% ----------------------------------------------------
global LM;
LM = zeros(NumElem,NumGL*NumNosEl);
for e=1:NumElem
    in = [1 2 3];
    Incide = Incid(e,:);
    for j=1:NumNosEl
        J=Incide(j);
        LM(e,in)=ID(J,:);
        in = in + 3;
    end
end

% Incializa vetores
KT      = zeros(NumGLTot,NumGLTot);
F       = zeros(NumGLTot,1);
U       = zeros(NumGLTot,1);
R       = zeros(NumGLTot,1);
dU      = zeros(NumGLTot,1);
Stress  = zeros(NumElem,NPInt*6);
Epl0    = zeros(NumElem,NPInt*6);
Epl1    = zeros(NumElem,NPInt*6);
Epleq0  = zeros(NumElem,NPInt);
Epleq1  = zeros(NumElem,NPInt);
DEpleq  = zeros(NumElem,NPInt);
Beta0  = zeros(NumElem,NPInt*6);
Beta1  = zeros(NumElem,NPInt*6);

% -----------------------------------------
% Calcula vetor de carga
for n = 1:NumCCNewmann
    No =    CCNewmann(n,1);
    gl =    CCNewmann(n,2);
    value = CCNewmann(n,3);
    eq    = ID(No,gl);
    F(eq) = F(eq) + value;
end;

% --------------------------------------------------
% Calcula Numero de substeps (incrementos de carga)
% --------------------------------------------------
incf = LoadSteps(1,1)/LoadSteps(1,3);
if (incf == 0)
    FFactor = 0;
else
    FFactor = (incf:incf:LoadSteps(1,1));
end
incu = LoadSteps(1,2)/LoadSteps(1,3);
if (incu == 0)
    UFactor = 0;
else
    UFactor = (incu:incu:LoadSteps(1,2));
end
cont = LoadSteps(1,3);
if (NumLoadSteps > 1)
    for ls = 2:NumLoadSteps
        incf = (LoadSteps(ls,1)-LoadSteps(ls-1,1))/LoadSteps(ls,3);
        incu = (LoadSteps(ls,2)-LoadSteps(ls-1,2))/LoadSteps(ls,3);
        ils = (cont+1:cont+LoadSteps(ls,3));
        cont = cont+LoadSteps(ls,3);
        if (incf == 0)
            FFactor(ils) = LoadSteps(ls-1,1);
        else
            FFactor(ils) = (LoadSteps(ls-1,1)+incf:incf:LoadSteps(ls,1));
        end
        
        if (incu == 0)
            UFactor(ils) = LoadSteps(ls-1,2);
        else
            UFactor(ils) = (LoadSteps(ls-1,2)+incu:incu:LoadSteps(ls,2));
        end
    end
end
NumLoadSubsteps = cont;

%DeltaT = t/NumLoadSubsteps;

% Incrementos de carga
for ls = 1:NumLoadSubsteps
    disp(ls) 
    if ls==15
        a=1
    end 
    %------------------------------------
    % Monta forcas externas do incremento
    FExt = F*FFactor(ls);
    
    %------------------------------------
    % Coloca condicoes de Dirichlet em U do subincremento
    for n = 1:NumCCDirichlet
        no=CCDirichlet(n,1);
        Gl=CCDirichlet(n,2);
        valor=CCDirichlet(n,3);
        eq    = ID(no,Gl);
        U(eq) = valor*UFactor(ls);
    end
    
    
    %------------------------------------
    % Inicia iteracoes de Newton
    Convergence = 0;
    NMaxIter = 1000;
    
    for k=1:NMaxIter
        % Calculo do Residuo e matriz tangente
        ComputeResidualKT();
        
%         %Calculo de Matriz tangente por diferencas finitas
%         [KTDF] = ComputeKTDF();
%         Dif = KT-KTDF;
%         NorKT = norm(KT,'fro');
%         NorDif = norm(Dif,'fro');
%         if (NorDif/NorKT > 1e-4
%             Erro = 0;
%         end
        
        % Introduz CCDirichlet homogenea no problema
        for n = 1:NumCCDirichlet
            no=CCDirichlet(n,1);
            Gl=CCDirichlet(n,2);
            eq    = ID(no,Gl);
            for g = 1:NumGLTot
                KT(eq,g) = 0;
                KT(g,eq) = 0;
            end
            KT(eq,eq) = 1;
            R(eq) = 0;
        end
        
        % Verificacao se residuo e satisfeito
        NormR = norm(R);
        if (k==1)&&(ls==1)
            NormR0 = NormR;
        end
        
        if ((NormR / NormR0) < MaxResidualError)
            Converged = 1;
            break;
        end
        
        % 5. Calculo de atualizacao de variaveis
        dU = - KT \ R;
        U = U + dU;
    end
    
    if (k == NMaxIter)
        fprintf('\n------------------------------------');
        fprintf('ERRO: Newton Global!!!!!');
        fprintf('------------------------------------');
        break
    end
    
    % Atualiza Vetores
    Epl0 = Epl1;
    Epleq0 = Epleq1;
    Beta0 = Beta1;
    
    
    % Auxiliar para plotagem
    Sz(ls+1) = Stress(1,3); %Elemento 1, tensao Sz
    Uz(ls+1) = U(15); % No 5 x 3GL
end
figure(1)
Sz(1) = 0;
Uz(1) = 0;
plot (Uz,Sz);
ylabel({'Tensão [N/mm]'});
xlabel({'Deslocamento [mm]'});
grid on
axis square 
hold on
%
% % -------------------------------------------
% % Gera malha para plotagem da configuracao
% %   deformadada
% % -------------------------------------------
% hold on;
% CoordDef=zeros(NumNos,2);
% Uf=zeros(NumNos*NumGL,1);
%
% c=1;
% b=1;
% for i=1:NumNos
%    for j=1:NumGL
%       if ID(i,j) == 0
%          Uf(c)=0;
%       else
%          Uf(c)=U(b);
%          b=b+1;
%       end
%       c=c+1;
%    end
% end
%
% plot(Coord,Uf,'-r');
