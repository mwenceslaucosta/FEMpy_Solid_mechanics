function [DFD] = GetDFD(Etot, EplIp0, EpleqIp0)
for i = 1:6
    Etopf = Etot;
    Etopf(i) = Etopf(i) + 1e-8;
    [Stressf, Eplf, Epleqf, DEpleqip] = GetStress(Etopf, EplIp0, EpleqIp0);

    Etopa = Etot;
    Etopa(i) = Etopa(i) - 1e-8;
    [Stressa, Epla, Epleqa, DEpleqip] = GetStress(Etopa, EplIp0, EpleqIp0);

    DStress = (Stressf - Stressa)/2e-8;
    DFD(i,:)=DStress;
end

    