function [Sc, Sb] = getSbSc(P,Q,X,k1,k2,distType)
[d,N] = size(X);
Sc = 0;
Sb = 0;

distvec = pdist(X',distType);
dist = squareform(distvec);

sparseP = sparse(N,N);
sparseQ = sparse(N,N);

for i = 1:N
    sparseP(i,P(i,:)) = 1;
    sparseQ(i,Q(i,:)) = 1;
end

Sc = full(1/N/k1 * sum(sum((sparseP .* (dist.^2)))));
Sb = full(1/N/k2 * sum(sum((sparseQ .* (dist.^2)))));
