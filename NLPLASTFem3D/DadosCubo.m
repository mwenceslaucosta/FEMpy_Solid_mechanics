% Dados para cubo de 1 elemento 

% Numero de GL por no
NumGL = 3;

% Numero de nos do elemento 
NumNosEl = 8;

% N?mero de pontos de integracao
NPInt = 8;

% Numero de nos (cx, cy)
Coord = [  0.0   0.0    0.0
           1.0   0.0    0.0
           1.0   1.0    0.0
           0.0   1.0    0.0
          
           0.0   0.0    1.0
           1.0   0.0    1.0
           1.0   1.0    1.0
           0.0   1.0    1.0 ];
          
%           0.0  0.0  4.0
%           1.0  0.0  4.0
%           1.0  1.0  4.0
%           0.0  1.0  4.0  ];       
NumNos = size(Coord,1);


% Numero de elementos (incidencia)
Incid = [ 1 2 3 4 5 6 7 8 ];
%           5 6 7 8 9 10 11 12];
tam = size(Incid);
NumElem = tam(1);

% Propriedades de Material
%          E     Nu     Sy   HB    K1    K2    theta
PropMat = [2000  0.25    20   50    20     10     40]
 
% Numero de Condicoes de Contorno de Dirichlet
% <no> <GL> <valor>
CCDirichlet = [1 3 0
               2 3 0 
               3 3 0
               4 3 0
               1 1 0 
	           4 1 0 
               1 2 0
               2 2 0 
               5 3 1.0 
	           6 3 1.0  
	           7 3 1.0 
               8 3 1.0]; 
tam = size(CCDirichlet);
NumCCDirichlet = tam(1);
 

% Numero de Condicoes de Contorno de Newmann
% <no> <GL> <valor>
CCNewmann =   [];
%                [5 3 100
%                 6 3 100
%                 7 3 100
%                 8 3 100];
tam = size(CCNewmann);
NumCCNewmann = tam(1);

% Forca de corpo (constante) <gl>  <valor>
BodyForce = [ 1  0.0
   		      2  0.0
              3  0.0 ];

% Neumann Load History (<Fator Neumann> <Fator Dirichlet> <numsubstep>
LoadSteps = [1.0   0.04  60
             1.0  -0.04  60
             1.0   0.04  60];
         
tam = size(LoadSteps);
NumLoadSteps = tam(1);


% Erro de Convergencia em equilibrio
MaxResidualError = 1e-6;
         
         
      