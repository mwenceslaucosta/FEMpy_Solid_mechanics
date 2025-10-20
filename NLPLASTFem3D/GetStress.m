function [StressIp, EplIp1, EpleqIp1, DEpleqIp, BetaIp1] = GetStress(Etot, EplIp0, EpleqIp0, BetaIp0)
           
global PropMat;
Young = PropMat(1);
Mu = PropMat(2);
Sy0 = PropMat(3);
HB = PropMat(4);
K1 = PropMat(5);
K2 = PropMat(6);
theta = PropMat(7);

G = Young/(2*(1+Mu));
Bulk = Young/(3*(1-2*Mu));

% Deformacao Elastica trial
EelTr = Etot' - EplIp0;
% Auxiliar
i1 = [1 2 3];
i2 = [4 5 6];
sqrt32 = 1.224744871391589;

% Deformacao volumetrica
Eelv = sum(EelTr(i1));
Eelv3 = Eelv / 3;

% Pressao
P = Bulk * Eelv;

% Deformacao deviatorica trial
EdevTr = EelTr;
EdevTr(i1) = EdevTr(i1) - Eelv3;

% Tensao deviatorica trial
STr = EdevTr * G;
STr(i1)  = 2*STr(i1);
DeltaTr = STr - BetaIp0;

% Tensao equivalente trial 
% O esquema aqui e entender que o produto interno dos tensores tensao
% possuem um termo que se repete em todos os elementos fora da diagonal
% principal. Por isto Str(i1)*Str(i1)' aparece uma vez na norma, e Str(i2)*Str(i2)'
% aparece duas vezes.
NormDeltaTr = sqrt((DeltaTr(i1)*DeltaTr(i1)' + 2*DeltaTr(i2)*DeltaTr(i2)'));
qTr = NormDeltaTr * sqrt32;

% Verifica viabilidade elastica (H = K -> Encruamento isotropico)
Sy = Sy0 + K1*EpleqIp0 + K2*(1-exp(-theta*EpleqIp0));
ftr = qTr - Sy;
if ftr <= 0
    % Passo elastico
    StressIp = STr + [P P P 0 0 0]; % Soma a parte volumetrica pois nao ha adicao de parcela deviatorica (nao ha plasticidade)
    DEpleqIp = 0;  
    EplIp1 = EplIp0;
    EpleqIp1 = EpleqIp0;
    BetaIp1 = BetaIp0;
    return; 
end

% Passo plastico
% Caso o comportamento seja isotropico linear, tem-se;
% DEpleqIp = ftr/(3*G + H); 
% Caso o comportamento seja isotropico nao linear,
tol = 1e-5;
NumIt_Newton = 10000000;
DEpleqIp0 = 0;
EpleqIp1 = EpleqIp0 + DEpleqIp0;
for k = 1:1: NumIt_Newton
    
    f =  qTr - DEpleqIp0*(3*G + HB) - (Sy0 + K1*EpleqIp1 + K2*(1-exp(-theta*EpleqIp1)));
    dSy0_dEpleq1 = K1 + K2*theta*exp(-theta*EpleqIp1);
    df_dDEpleqIp0 = -(3*G + HB + dSy0_dEpleq1);
    
    Delta_DEpleqIp0 = - f/df_dDEpleqIp0;
    
    DEpleqIp0 = DEpleqIp0 + Delta_DEpleqIp0;
    
    EpleqIp1 = EpleqIp1 + DEpleqIp0;
    if f <= tol
        fprintf('\n Iteracao Newton bem sucedida')  
        break    
    end
    
end

DEpleqIp = DEpleqIp0 + Delta_DEpleqIp0;

% Atualiza outras variaveis
Normal = sqrt32 * DeltaTr/NormDeltaTr;
Normal(i2) = Normal(i2)*2;
EplIp1 = EplIp0 + DEpleqIp*Normal;

% Calcula tensao deviatorica
S = STr - 2*G*DEpleqIp*Normal;  
StressIp = S + [P P P 0 0 0];
BetaIp1 = BetaIp0 + (2/3)*HB*DEpleqIp*Normal;
EpleqIp1 = EpleqIp0 + DEpleqIp;











