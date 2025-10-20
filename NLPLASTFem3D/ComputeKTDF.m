% ---------------------------------------------------------
% Funcao que retorna residuo de elemento de Barra 1D Linear
% ---------------------------------------------------------

function [KTDF] = ComputeKTDF()
GetGlobal;
for i = 1:24
    Uori = U;
    Rori = R;
    
    U(i) = Uori(i) + 1e-9;
    ComputeResidualKT();
    Rf = R;

    U(i) = Uori(i) - 1e-9;
    ComputeResidualKT();
    Ra = R;
    
    DR = (Rf - Ra)/(2e-9);
    KTDF(i,:)=DR;
end
U = Uori;
R = Rori;



