function [Xt_test_new_classify,W,B,net,Xs_new,Xt_test_new] = Ylabel_Lowranknorm_of_PRES_DTML_train(Xs,Labels,opts,Xt_test)
[d, Ns] = size(Xs);
Xt=Xt_test;
Nt=size(Xt,2);
C=size(Labels,1);
%%%%%% Initial W and b %%%%%%%%%%
inputD = d;
Z=eye(Ns,Nt);

R2 = zeros(Ns,Nt);
alphaJ=opts.gamma2 ;
rho = 10;
max_mu = 1e6;
mu = 0.3;
J=zeros(Ns,Nt);

Xs_new =[];
Xt_test_new=[];
for m = 1:opts.M %逐层前向传播
    hidNum = opts.hidNum(m);
    net.layer{m}.W = eye(hidNum,inputD);
    net.layer{m}.b = zeros(hidNum,1);
    inputD = hidNum;   
end
%%%%%% Initial last W and B%%%%%%%%%%%%%
   W = eye(C,inputD);
   B = zeros(C,1);
%%%%%% End Initial %%%%%%%%%%%%%

%%%%%% End getting P and Q %%%%%
Dts = cell(1,opts.M);

%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%
lastJcost = inf;
  Dts_curve =[];
  Dclassify_curve =[];
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
%          inputLayer_s = inputLayer_s./repmat(sqrt(sum(inputLayer_s.^2)),[size(inputLayer_s,1) 1]);  %//归一化
%          inputLayer_t = inputLayer_t./repmat(sqrt(sum(inputLayer_t.^2)),[size(inputLayer_t,1) 1]);  %//归一化
%          inputLayer_test = inputLayer_test./repmat(sqrt(sum(inputLayer_test.^2)),[size(inputLayer_test,1) 1]);  %//归一化
    end 
        
      [hs{opts.M+1},zs{opts.M+1}] =  PRESDTML_forward(inputLayer_s,W,B,opts);
%        zs{opts.M+1} = zs{opts.M+1}./repmat(sqrt(sum(zs{opts.M+1}.^2)),[size(zs{opts.M+1},1) 1]);  %//归一化
      if strcmp(opts.distType,'softmax')  %'euclidean'     
           zs{opts.M+1} = exp(bsxfun(@minus, zs{opts.M+1}, max(zs{opts.M+1},[],2)));
           zs{opts.M+1} = bsxfun(@rdivide, zs{opts.M+1}, sum(zs{opts.M+1}, 2)); 
      end    
    clear inputLayer_s inputLayer_t inputLayer_test;
    Dts{opts.M} = PRESgetDts(hs{opts.M},ht{opts.M},Z);
    Dclassify_s{opts.M} = PRESgetDclassify(zs{opts.M+1},Labels,opts);
    
     % updating Z
    Lz=opts.beta1*hs{opts.M}'* (hs{opts.M}*Z-ht{opts.M})+mu*(Z-J)/2+R2/2;
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
    
%       updating w,b,W,B
    for m = opts.M:-1:1
        if m == opts.M
            Lt = PREScomputeLastLt(hs{m},ht{m},zt{m},opts.actfuncType,Z);
            Ls = PREScomputeLastLs(hs{m},ht{m},zs{m},opts.actfuncType,Z); 
            LYs = PREScomputeLastLYs(W,zs{m+1},zs{m},opts.actfuncType,Labels);  
        else
            Lt = computeLsLt(net.layer{m+1}.W,lastLt,zt{m},opts.actfuncType);
            Ls = computeLsLt(net.layer{m+1}.W,lastLs,zs{m},opts.actfuncType);
            LYs = computeLsLt(net.layer{m+1}.W,lastLYs,zs{m},opts.actfuncType);
        end
        if m == 1
             [dJdW{m},dJdB{m}] = Ylabel_PRESgetDelta(Lt,Ls,LYs,net.layer{m}.W,net.layer{m}.b,Xs,Xt,opts,Z);
%             [dJdW{m},dJdB{m}] = PRESgetDelta(Lt,Ls,net.layer{m}.W,net.layer{m}.b,Xs,Xt,opts,Z);
        else
              [dJdW{m},dJdB{m}] = Ylabel_PRESgetDelta(Lt,Ls,LYs,net.layer{m}.W,net.layer{m}.b,hs{m-1},ht{m-1},opts,Z);
%             [dJdW{m},dJdB{m}] = PRESgetDelta(Lt,Ls,net.layer{m}.W,net.layer{m}.b,hs{m-1},ht{m-1},opts,Z);
            
        end 
        lastLs = Ls;
        lastLt = Lt;
        lastLYs = LYs;
   end
    
    for m = 1:opts.M
        net.layer{m}.W = net.layer{m}.W - opts.lr * dJdW{m}/norm(dJdW{m},2);
        net.layer{m}.b = net.layer{m}.b - opts.lr * dJdB{m}/norm(dJdW{m},2);
%         net.layer{m}.W = net.layer{m}.W - opts.lr * dJdW{m};
%         net.layer{m}.b = net.layer{m}.b - opts.lr * dJdB{m};
    end
 %     % updating W,B
 
     Lw=opts.beta2*(zs{opts.M+1}-Labels)* hs{opts.M}'+opts.gamma1*W ;
%       W=W-opts.lr * Lw;
     W=W-opts.lr * Lw/norm( Lw,2);

     
     Lb=opts.beta2*mean((zs{opts.M+1}-Labels),2)+opts.gamma1*B ;
%        B=B-opts.lr * Lb;
     B=B-opts.lr * Lb/norm(Lb,2);


       
%     % updating lr
%     opts.lr = opts.lr * opts.lr_decay;

      % updating R2
    R2 = R2+(mu*(Z-J));
    
    % updating mu
    mu = min(rho*mu,max_mu);
    
%     [Jcost,Dts1,normValue,normValue_Z] = PRESgetJcost(Dts{opts.M},net,opts,Z);
     [Jcost,Dts1,Dclassify_s1,normValue,normValue_Z] = Ylabel_PRESgetJcost(Dts{opts.M}, Dclassify_s{opts.M},net,opts,Z,W,B);
    if abs(Jcost - lastJcost) < opts.epsilon
        break;
    end
    lastJcost = Jcost;
%     fprintf('%d/%d, Jcost = %f, Dts = %f, Dclassify_s = %f,norm = %f, norm_Z = %f\n',k,opts.T,Jcost,Dts1,Dclassify_s1,normValue,normValue_Z);
    Dts_curve(k)=Dts1;
    Dclassify_curve(k) =Dclassify_s1;
end
%       figure(3);
%        plot(Dts_curve); 
%         hold on; 
%        plot(Dclassify_curve); 
%       legend( 'Dts','Dclassify');
%       xlabel('Iteration','FontName','Times New Roman','FontSize',50);
%       ylabel('Objective value','FontName','Times New Roman','FontSize',50);
%       set(gca,'FontName','Times New Roman','FontSize',15);
%       hold off;
%       title('Convergence','FontName','Times New Roman','FontSize',20);

      

%       figure(4);
%       imagesc(Z);
%       title('structure_Z');
  Xs_new = hs{opts.M};
  Xt_test_new = ht_test{opts.M};
  
  [hs{opts.M+1},Xt_test_new_classify] =  PRESDTML_forward(Xt_test_new,W,B,opts);
