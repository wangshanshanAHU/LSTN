function [Jcost,Dts1,normValue,normValue_Z] = PRESgetJcost(Dts,net,opts,Z)
% Jcost = Sc - opts.alpha * Sb + opts.beta * Dts;
Jcost =  0;
% Jcost = Dts;
Dts1= opts.beta1 * Dts;
normValue = 0;
% normValue_Z=opts.alpha * norm(Z,'fro')^2;
[U,sigma,V] = svd(Z,'econ');
normValue_Z = opts.gamma2 *sum(diag(sigma));
for m = 1:opts.M
    normValue = normValue + opts.gamma1 *((norm(net.layer{m}.W,'fro')^2 + norm(net.layer{m}.b,2)^2));
end
Jcost =Dts1 + normValue+ normValue_Z;