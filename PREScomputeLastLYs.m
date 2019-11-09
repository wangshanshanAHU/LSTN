function LsLt = PREScomputeLastLYs(W,zslast,zs,actfuncType,Y)
[C,Ns] = size(zslast);
[C,d] = size(W);
LsLt = zeros(d,Ns);
% tmp = 1/Ns * sum(hs,2) - 1/Nt * sum(ht,2);
% dev_zs = devActfunc(zs,actfuncType);
% LsLt = repmat(tmp,1,size(dev_zs,2)) .* dev_zs;
%  if strcmp(opts.distType,'softmax')  %'euclidean'     
%      zslast = exp(bsxfun(@minus, zslast, max(zslast,[],2)));
%      zslast = bsxfun(@rdivide, zslast, sum(zslast, 2)); 
%   end  
tmp = W'*(zslast-Y);
dev_zs = devActfunc(zs,actfuncType);
LsLt = tmp .* dev_zs;
