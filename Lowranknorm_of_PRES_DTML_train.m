function [Xt_train_new,net,Xs_new,Xt_test_new] = Lowranknorm_of_PRES_DTML_train(Xs,opts,Xt_test)
Xt=Xt_test;
[d, Ns] = size(Xs);
Nt=size(Xt,2);
%%%%%% Initial W and b %%%%%%%%%%
inputD = d;
Z=zeros(Ns,Nt);

R2 = zeros(Ns,Nt);
alphaJ=opts.gamma2 ;
rho = 1.01;
max_mu = 1e6;
mu = 0.3;
J=zeros(Ns,Nt);

Xs_new =[];
Xt_train_new=[];
Xt_test_new=[];
tic
for m = 1:opts.M %逐层前向传播
    hidNum = opts.hidNum(m);
    net.layer{m}.W = eye(hidNum,inputD);
    net.layer{m}.b = zeros(hidNum,1);
    inputD = hidNum;   
end
%%%%%% End Initial %%%%%%%%%%%%%

%%%%%% End getting P and Q %%%%%
Dts = cell(1,opts.M);

%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%
lastJcost = inf;
  Dts_curve =[];
  Obj_curve=[];
for k = 1:opts.T
    inputLayer_s = Xs;
    inputLayer_t = Xt;
    inputLayer_test = Xt_test;
    hs = cell(1,opts.M);
    ht = cell(1,opts.M);
    zs = cell(1,opts.M);
    zt = cell(1,opts.M);
    dJdW = cell(1,opts.M);
    dJdB = cell(1,opts.M);
    ht_test = cell(1,opts.M);
    zt_test = cell(1,opts.M);
    
    for m = 1:opts.M %逐层前向传播
         [hs{m},zs{m}] =  PRESDTML_forward(inputLayer_s,net.layer{m}.W,net.layer{m}.b,opts);
         [ht{m},zt{m}] =  PRESDTML_forward(inputLayer_t,net.layer{m}.W,net.layer{m}.b,opts);
          [ht_test{m},zt_test{m}] = PRESDTML_forward(inputLayer_test,net.layer{m}.W,net.layer{m}.b,opts);
        inputLayer_s = hs{m};
        inputLayer_t = ht{m};
         inputLayer_test = ht_test{m};
    end
    clear inputLayer_s inputLayer_t;
    Dts{opts.M} = PRESgetDts(hs{opts.M},ht{opts.M},Z);
    
     % updating Z
    Lz=hs{opts.M}'* (hs{opts.M}*Z-ht{opts.M})+mu*(Z-J)/2+R2/2;
%      Z=Z-opts.lr * Lz;
     Z=Z-opts.lr * Lz/norm(Lz,2);
     
       % updating J
    temp = Z + R2/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>alphaJ/mu));
    if svp>=1
        sigma = sigma(1:svp)-alphaJ/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    
%       updating W,B
    for m = opts.M:-1:1
        if m == opts.M
            Lt = PREScomputeLastLt(hs{m},ht{m},zt{m},opts.actfuncType,Z);
            Ls = PREScomputeLastLs(hs{m},ht{m},zs{m},opts.actfuncType,Z);            
        else
            Lt = computeLsLt(net.layer{m+1}.W,lastLt,zt{m},opts.actfuncType);
            Ls = computeLsLt(net.layer{m+1}.W,lastLs,zs{m},opts.actfuncType);
        end
        if m == 1
            [dJdW{m},dJdB{m}] = PRESgetDelta(Lt,Ls,net.layer{m}.W,net.layer{m}.b,Xs,Xt,opts,Z);
        else
            [dJdW{m},dJdB{m}] = PRESgetDelta(Lt,Ls,net.layer{m}.W,net.layer{m}.b,hs{m-1},ht{m-1},opts,Z);
        end 
        lastLs = Ls;
        lastLt = Lt;
    end
    
    for m = 1:opts.M
%         net.layer{m}.W = net.layer{m}.W - opts.lr * dJdW{m}/norm(dJdW{m},2);
%         net.layer{m}.b = net.layer{m}.b - opts.lr * dJdB{m}/norm(dJdW{m},2);
        net.layer{m}.W = net.layer{m}.W - opts.lr * dJdW{m};
        net.layer{m}.b = net.layer{m}.b - opts.lr * dJdB{m};
    end
       
%     % updating lr
%     opts.lr = opts.lr * opts.lr_decay;

      % updating R2
    R2 = R2+(mu*(Z-J));
    
    % updating mu
    mu = min(rho*mu,max_mu);
    
    [Jcost,Dts1,normValue,normValue_Z] = PRESgetJcost(Dts{opts.M},net,opts,Z);
%     if abs(Jcost - lastJcost) < opts.epsilon
%         break;
%     end
    lastJcost = Jcost;
    fprintf('%d/%d, Jcost = %f, Dts = %f, norm = %f, norm_Z = %f\n',k,opts.T,Jcost,Dts1,normValue,normValue_Z);
    Dts_curve(k)=Dts1;
    Obj_curve(k)=Jcost;       
end
% toc
%       figure(1);
%        hold on;        
%        plot(Obj_curve);
%        plot(Dts_curve); 
% %       xlabel('Iteration','FontName','Times New Roman','FontSize',50);
% %       ylabel('Dts Objective value','FontName','Times New Roman','FontSize',50);
% %       set(gca,'FontName','Times New Roman','FontSize',15);
%       legend( 'Obj_m_i_n','Dst_m_i_n');
%       hold off;
%       title('Convergence');
% %       title('Convergence','FontName','Times New Roman','FontSize',20);          
%       figure(2);
%       imagesc(Z);
%      title('structure_Z');
     
  Xs_new = hs{opts.M};
  Xt_train_new = ht{opts.M};
  Xt_test_new = ht_test{opts.M};
