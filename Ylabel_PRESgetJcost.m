function [Jcost,Dts,Dclassify_s,normValue,normValue_Z] = Ylabel_PRESgetJcost(Dts,Dclassify_s,net,opts,Z,W,B)
% Jcost = Sc - opts.alpha * Sb + opts.beta * Dts;
Jcost =  0;
Jcost =  Dts;
normValue = 0;
% normValue_Z=opts.alpha * norm(Z,'fro')^2;
[U,sigma,V] = svd(Z,'econ');
normValue_Z = opts.gamma2 *sum(diag(sigma));
for m = 1:opts.M
    normValue = normValue + opts.gamma1 *((norm(net.layer{m}.W,'fro')^2 + norm(net.layer{m}.b,2)^2));
end
normValue = normValue + opts.gamma1 *((norm(W,'fro')^2 + norm(B,2)^2));
Jcost =opts.beta1 * Jcost +opts.beta2*Dclassify_s+ normValue+ normValue_Z;