
function [B,DetJ] = GetBdetJ(Pint, CoordEl)

umePint = (1 - Pint);
umaPint = (1 + Pint);

umacsi = umaPint(1);
umaeta = umaPint(2);
umazet = umaPint(3);

umecsi = umePint(1);
umeeta = umePint(2);
umezet = umePint(3);

dpcsi0 = -umeeta*umezet;
dpcsi1 = -dpcsi0;
dpcsi2 =  umaeta*umezet;
dpcsi3 = -dpcsi2;
dpcsi4 = -umeeta*umazet;
dpcsi5 = -dpcsi4;
dpcsi6 =  umaeta*umazet;
dpcsi7=  -umaeta*umazet;

dpeta0= -umecsi*umezet;
dpeta1= -umacsi*umezet;
dpeta2=  umacsi*umezet;
dpeta3=  umecsi*umezet;
dpeta4= -umecsi*umazet;
dpeta5= -umacsi*umazet;
dpeta6=  umacsi*umazet;
dpeta7=  umecsi*umazet;

dpzet0= -umecsi*umeeta;
dpzet1= -umacsi*umeeta;
dpzet2= -umacsi*umaeta;
dpzet3= -umecsi*umaeta;
dpzet4=  umecsi*umeeta;
dpzet5=  umacsi*umeeta;
dpzet6=  umacsi*umaeta;
dpzet7=  umecsi*umaeta;

DPsiDCsi = [dpcsi0 dpcsi1 dpcsi2 dpcsi3 dpcsi4 dpcsi5 dpcsi6 dpcsi7
            dpeta0 dpeta1 dpeta2 dpeta3 dpeta4 dpeta5 dpeta6 dpeta7
            dpzet0 dpzet1 dpzet2 dpzet3 dpzet4 dpzet5 dpzet6 dpzet7]*0.125;

J = DPsiDCsi*CoordEl;
DetJ = det(J);

JInv(1,1)=J(2,2)*J(3,3) - J(2,3)*J(3,2);
JInv(1,2)=J(1,3)*J(3,2) - J(1,2)*J(3,3);
JInv(1,3)=J(1,2)*J(2,3) - J(1,3)*J(2,2);
JInv(2,1)=J(2,3)*J(3,1) - J(2,1)*J(3,3);
JInv(2,2)=J(1,1)*J(3,3) - J(1,3)*J(3,1);
JInv(2,3)=J(1,3)*J(2,1) - J(1,1)*J(2,3);
JInv(3,1)=J(2,1)*J(3,2) - J(2,2)*J(3,1);
JInv(3,2)=J(1,2)*J(3,1) - J(1,1)*J(3,2);
JInv(3,3)=J(1,1)*J(2,2) - J(1,2)*J(2,1);

JInv = JInv*(1/DetJ);

DPsiDX = JInv*DPsiDCsi;

B=zeros(6,24);
for jj = 1:8
    j = jj-1;
    dpsidx = DPsiDX(1,jj);
    B(1,j*3+1) = dpsidx;
    B(4,j*3+2) = dpsidx;
    B(6,j*3+3) = dpsidx;    
    
    dpsidy = DPsiDX(2,jj);
    B(2,j*3+2) = dpsidy;
    B(4,j*3+1) = dpsidy;
    B(5,j*3+3) = dpsidy;    
  
    dpsidz = DPsiDX(3,jj);
    B(3,j*3+3) = dpsidz;
    B(5,j*3+2) = dpsidz;
    B(6,j*3+1) = dpsidz;    
end




