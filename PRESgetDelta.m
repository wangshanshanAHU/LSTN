function [dJdW,dJdB] = getDelta(Lt,Ls,W,b,hs,ht,opts,Z)
% alpha = opts.alpha;
% beta = opts.beta;
% gamma = opts.gamma;

[~,N] = size(hs);
% [~,Ns_mmd] = size(Ls_mmd);
% [~,Nt_mmd] = size(Lt_mmd);


% tmpPW = zeros(size(W));
% tmpQW = tmpPW;
% tmpPB = zeros(size(b));
% tmpQB = zeros(size(b));
% for i = 1:N
%     for j = 1:size(P(i,:),2)
%         tmpPW = tmpPW +  L{i}(:,P(i,j)) * hs(:,i)' + L{P(i,j)}(:,i) * hs(:,P(i,j))';
%         tmpPB = tmpPB + L{i}(:,P(i,j)) + L{P(i,j)}(:,i);
%     end
%     
%     for j = 1:size(Q(i,:),2)
%         tmpQW = tmpQW + L{i}(:,Q(i,j))*hs(:,i)' + L{Q(i,j)}(:,i) * hs(:,Q(i,j))';
%         tmpQB = tmpQB + L{i}(:,Q(i,j)) +  L{Q(i,j)}(:,i);
%     end
% end
% tmpLtW = zeros(size(W));
% tmpLsW = zeros(size(W));
% for i = 1:Ns_mmd   
%     tmpLsW = tmpLsW + Ls_mmd(:,i)*hs(:,i)';
% end
% for i = 1:Nt_mmd
%     tmpLtW = tmpLtW + Lt_mmd(:,i)*ht(:,i)';
% end
% % dJdW = 2/(N*k1) * tmpPW - 2*alpha/(N*k2)*tmpQW + 2*beta/Nt*tmpLtW + 2*beta/Ns*tmpLsW + 2*gamma * W;
% dJdW =  2*Ls*hs' -2*Lt*ht'+ 2*beta/Nt_mmd*tmpLtW + 2*beta/Ns_mmd*tmpLsW + 2*gamma * W;
meanLt = mean(Lt,2);
meanLs = mean(Ls,2);
% sumLt_mmd = sum(Lt_mmd,2);
% sumLs_mmd = sum(Ls_mmd,2);
% % dJdB = 2/(N*k1)*tmpPB - 2*alpha/(N*k2)*tmpQB + 2*beta/Nt*sumLt + 2*beta/Ns*sumLs + 2*gamma * b;
% dJdB = 2*meanLs- 2*meanLt+ 2*beta/Nt_mmd*sumLt_mmd + 2*beta/Ns_mmd*sumLs_mmd+ 2*gamma * b;
dJdW =  2*opts.beta1*Ls*hs' -2*opts.beta1*Lt*ht'+  2*opts.gamma1 * W;
dJdB = 2*opts.beta1*meanLs- 2*opts.beta1*meanLt+ 2*opts.gamma1 * b;


