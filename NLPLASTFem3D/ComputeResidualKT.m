% ---------------------------------------------------------
% Funcao que retorna residuo e Matriz Tg de Hexa8
% ---------------------------------------------------------

function ComputeResidualKT()

GetGlobal;
R = R*0.0;
KT = KT*0.0;

% Pontos de integracao do elemento
a = 1/sqrt(3);
PInt = [-a -a -a; a -a -a; a  a -a; -a  a -a
        -a -a  a; a -a  a; a  a  a;-a  a  a ];
WInt = 1;

% 2. Dimensiona vetores de elemento
% UEl     = zeros(24,1);
% LMEl    = zeros(24,1);
% CoordEl = zeros(8,3);
% IncidEl = zeros(8);
KTEl    = zeros(24,24);
FIntEl  = zeros(24,1);

% 4. Inicializa Residuo
R = - FExt;
    
% 5. Loop sobre elementos
ie = 1:24;
for e = 1:NumElem

    %5. Captura valores elementares
    IncidEl = Incid(e,:);
    CoordEl = Coord(IncidEl,:);
    LMEl = LM(e,:);
    UEl = U(LMEl);
    FIntEl(ie) = 0;
    KTEl(ie,ie) = 0;    
    
    %6. Loop sobre pontos de integracao
    in = 1:6;
    for ip = 1:NPInt
        
        %inicializa contador e variaveis
        EpleqIp0 = Epleq0(e,ip);
        EplIp0  = Epl0(e,in);
        BetaIp0 = Beta0(e,in);
        pint = PInt(ip,:);
        
        %Calcula deformacao total e trial
        [B,DetJ] = GetBdetJ(pint, CoordEl);
        Etot = B*UEl;
        if ip==1
           disp(Etot)
        end 
        
        % Calcula tensao e matriz elastoplastica
        [StressIp, EplIp1, EpleqIp1, DEpleqIp, BetaIp1] = GetStress(Etot, EplIp0, EpleqIp0, BetaIp0);
        [D] = GetD(StressIp, DEpleqIp, BetaIp1, EpleqIp1);
        
%         [DFD] = GetDFD(Etot, EplIp0, EpleqIp0);
%          
%         DifD = DFD - D;
%         if (norm(DifD,'fro') > 1e-4)
%              r = 0;
%         end
         
        % Insere forca interna e rigidez nas matrizes
        FIntEl = FIntEl + B'*StressIp'*DetJ*WInt;
        KTEl = KTEl + B'*D*B*DetJ*WInt;   
        
        % Coloca valores do IP no vetor
        Epleq1(e,ip) = EpleqIp1;
        Epl1(e,in) = EplIp1;
        DEpleq(e,ip) = DEpleqIp;
        Stress(e,in) = StressIp;
        Beta1(e,in) = BetaIp1;
     
        %Atualiza contador
        in = in+6;
    end
    in = 1:24;
    R(LMEl) = R(LMEl) + FIntEl(in);
    KT(LMEl,LMEl) = KT(LMEl,LMEl)+ KTEl;
end




      



