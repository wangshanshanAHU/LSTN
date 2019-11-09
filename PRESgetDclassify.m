function Dts = PRESgetDclassify(Xs,Xt,opts)
[d,Nt] = size(Xt);
[d,Ns] = size(Xs);
y=Xt';
a=Xs';
% Dts_mmd =opts.beta* norm((sum(Xt,2)./Nt-sum(Xs,2)./Ns).^2,2)^2;
% Dts = norm(Xs*Z-Xt,'fro')^2+Dts_mmd;
switch opts.distType 
        case 'euclidean'
           Dts = norm(Xs-Xt,'fro')^2;
        case 'softmax'
           Dts = -sum(sum(y .* log(a)));
end