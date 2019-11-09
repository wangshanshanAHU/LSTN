function [P,Q] = getPQ(Xs,Ls,k1,k2,distType)
[d,N] = size(Xs);
P = zeros(N,k1);
Q = zeros(N,k2);

distvec = pdist(Xs',distType);
dist = squareform(distvec);
[~,index] = sort(dist);
for i = 1:N
    vecPindex = zeros(k1,1);
    indexP = 1;
    vecQindex = zeros(k2,1);
    indexQ = 1;
    for j = 1:N
        if indexP > k1 && indexQ > k2
            break;
        end
        if Ls(j) == Ls(i)
            if indexP > k1 || j == i
                continue;
            end
            vecPindex(indexP) = index(j,i);
            indexP = indexP + 1;
            
        else
            if indexQ > k2 || j == i
                continue;
            end
            vecQindex(indexQ) = index(j,i);
            indexQ = indexQ + 1;
         end
    end
     P(i,:) = vecPindex;
     Q(i,:) = vecQindex;
end