global NumElem; % Numero de Elementos
global NumNos;  % Numero de Nos
global NPInt;   % Numero de Pontos de Integracao no Elemento
global Coord;   % Vetor de Coordenadas
global Incid;   % Vetor de Conectividade ou Incidencia
global LM;      % Vetor de indexacao do elemento na matriz Global
global PropMat; % Vetor com Propriedades de Material
global ID;      % Matriz de graus de liberdade
global NumGLTot;% Numero total de Graus de Liberdade

global KT;      % Matriz Tangente
global F;       % Vetor de Forca 
global FExt;    % Vetor de Forca externa
global FInt;    % Vetor de Forca interna
global U;       % Vetor de Deslocamento
global R;       % Vetor de Residuo
global dU;      % Vetor de Correcao de Deslocamento (Metodo de Newton)
global Stress;  % Vetor de Tensoes zeros(NumElem,NPInt*6);
global Epl0;    % Vetor de deformacao Plastica no instante t DIM(NumElem,NPInt*6);
global Epl1;    % Vetor de deformacao Plastica no instante t+1 DIM(NumElem,NPInt*6);
global Epleq0;  % Vetor de deformacao Plastica equivalente no instante t DIM(NumElem*NPInt);
global Epleq1;  % Vetor de deformacao Plastica equivalente no instante t+1 DIM(NumElem*NPInt);
global DEpleq;  % Vetor Incremento de deformacao Plastica equivalente DIM(NumElem*NPInt);
global Beta0;
global Beta1;

