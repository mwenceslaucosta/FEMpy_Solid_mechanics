function [D] = GetD(Stress, DEpleqIp, BetaIp1, EpleqIp1)

global PropMat;
Young = PropMat(1);
Mu = PropMat(2);
HB = PropMat(4);
K1 = PropMat(5);
K2 = PropMat(6);
theta = PropMat(7);

G = Young/(2*(1+Mu));
Bulk = Young/(3*(1-2*Mu));

% Auxiliar
i1 = [1 2 3];
i2 = [4 5 6];
R1D3 = 1/3;
R2D3 = 1-R1D3;
sqrt32 = 1.224744871391589;
G3 = 3*G;
G2 = 2*G;
DevPrj = [R2D3 -R1D3 -R1D3     0    0    0
         -R1D3  R2D3 -R1D3     0    0    0
         -R1D3 -R1D3  R2D3     0    0    0
             0     0     0   0.5    0    0
             0     0     0     0  0.5    0
             0     0     0     0    0  0.5];
SOI = [1 1 1 0 0 0];         
if (DEpleqIp > 0)
    % Matriz elastoplastica
    % Calcula pressao P e tensor deviatorico S
    P = sum(Stress(i1)) * R1D3;
    S = Stress;
    S(i1) = S(i1) - P;

    % Tensao equivalente
    NormS = sqrt((S(i1)*S(i1)' + 2*S(i2)*S(i2)'));
    q = NormS * sqrt32;

    % Tensao equivalente trial
    dSy0_dEpleq1 = K1 + K2*theta*exp(-theta*EpleqIp1);
    qTr = q + G3*DEpleqIp;
    a = G2*(1-G3*DEpleqIp/qTr);
    b = 6*G*G*(DEpleqIp/qTr- (1/(G3+HB+dSy0_dEpleq1)))/(NormS*NormS);
    D = a*DevPrj + b*(S'*S) + (SOI'*SOI)*Bulk;
else
    % Matriz elastica
    D = G2*DevPrj + (SOI'*SOI)*Bulk;
end
          











