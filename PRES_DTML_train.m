function [net,Xs_new,Xt_train_new,Xt_test_new] = RES_DTML_train(Xs,Labels,Xt,Labelt,opts,Xt_test)
[d, Ns] = size(Xs);
Nt=size(Xt,2);
%%%%%% Initial W and b %%%%%%%%%%
inputD = d;
Z=eye(Ns,Nt);
Xs_new =[];
Xt_train_new=[];
Xt_test_new=[];
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
for k = 1:opts.T
    inputLayer_s = Xs;
    inputLayer_t = Xt;
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
        inputLayer_s = hs{m};
        inputLayer_t = ht{m};
        if (k == opts.T) && (m == opts.M)
         [ht_test{m},zt_test{m}] = PRESDTML_forward(Xt_test,net.layer{m}.W,net.layer{m}.b,opts);
        end   
    end
    clear inputLayer_s inputLayer_t;
    Dts{opts.M} = getDts(hs{opts.M},ht{opts.M},Z);
    
     % updating Z
    Lz=hs{opts.M}'* (hs{opts.M}*Z-Xt)+opts.alpha*Z;
    Z=Z-opts.lr * Lz/norm(Lz,2);
    
    
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
        net.layer{m}.W = net.layer{m}.W - opts.lr * dJdW{m}/norm(dJdW{m},2);
        net.layer{m}.b = net.layer{m}.b - opts.lr * dJdB{m}/norm(dJdW{m},2);
    end
       
    % updating lr
%     opts.lr = opts.lr * opts.lr_decay;
    
    [Jcost,Dts1,normValue,normValue_Z] = PRESgetJcost(Dts{opts.M},net,opts,Z);
    if abs(Jcost - lastJcost) < opts.epsilon
        break;
    end
    lastJcost = Jcost;
    fprintf('%d/%d, Jcost = %f, Dts = %f, norm = %f, norm_Z = %f\n',k,opts.T,Jcost,Dts1,normValue,normValue_Z);
    Dts_curve(k)=Dts1;
end
      figure(2);
       plot(Dts_curve); 
       hold on;   
      xlabel('Iteration','FontName','Times New Roman','FontSize',50);
      ylabel('Dts Objective value','FontName','Times New Roman','FontSize',50);
      set(gca,'FontName','Times New Roman','FontSize',15);
      hold off;
      title('Convergence','FontName','Times New Roman','FontSize',20);
  Xs_new = hs{opts.M};
  Xt_train_new = ht{opts.M};
  Xt_test_new = ht_test{opts.M};
