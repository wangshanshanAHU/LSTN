function [Jcost,Dts,normValue,normValue_Z] = getJcost(Dts,net,opts,Z)
% Jcost = Sc - opts.alpha * Sb + opts.beta * Dts;
Jcost =  0;
Jcost =  Dts;
normValue = 0;
normValue_Z=opts.alpha * norm(Z,'fro')^2;
for m = 1:opts.M
    normValue = normValue + opts.gamma *((norm(net.layer{m}.W,'fro')^2 + norm(net.layer{m}.b,2)^2));
end
Jcost = Jcost +  normValue+ normValue_Z;