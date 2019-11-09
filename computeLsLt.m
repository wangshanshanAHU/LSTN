function LsLt = computeLsLt(W,lastLsLt,zszt,actfuncType)
[d,N] = size(zszt);
LsLt = zeros(d,N);
dev_zszt = devActfunc(zszt,actfuncType);
LsLt = (W' * lastLsLt) .* dev_zszt;
