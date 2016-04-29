for i = 1:20
    R = ReadTensor(sprintf('P-4-100-200.%d.txt', i));
    RR = round(R, 4);
    z = spdiag(1 ./ sum(RR, 1)');
    S = RR * z;
    genSymScript(S);
    Y = mySym(S);
    size(Y)
end
