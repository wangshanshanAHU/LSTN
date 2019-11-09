function Dts = PRESgetDts(Xs,Xt,Z)
[d,Nt] = size(Xt);
[d,Ns] = size(Xs);
% Dts_mmd =opts.beta* norm((sum(Xt,2)./Nt-sum(Xs,2)./Ns).^2,2)^2;
% Dts = norm(Xs*Z-Xt,'fro')^2+Dts_mmd;
Dts = norm(Xs*Z-Xt,'fro')^2;