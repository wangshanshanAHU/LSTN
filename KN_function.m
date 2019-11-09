function [Ks, KY_test] = KN_function(Kernel,Xs_train,Xt_test)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
 switch Kernel 
       case'linear'    
       kervar1=1.2;% free parameter
       kervar2=10;% no use 
       case  'gauss'
       kervar1=1.5;% free parameter
       kervar2=10;% no use
 end  
       X=[Xs_train,Xt_test];  
       X = X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]);  %//归一化
       K    = gram(X',X',Kernel,kervar1,kervar2);
       K =max(K,K');
       K = K./repmat(sqrt(sum(K.^2)),[size(K,1) 1]);  %//归一化
%        Kt   = gram(X',Xt_train',Kernel,kervar1,kervar2);
%        Kt   = Kt./repmat(sqrt(sum(Kt.^2)),[size(Kt,1) 1]);  %//归一化
       Ks   = gram(X',Xs_train',Kernel,kervar1,kervar2);
       Ks   = Ks./repmat(sqrt(sum(Ks.^2)),[size(Ks,1) 1]);  %//归一化
       KY_test  = gram(X',Xt_test',Kernel,kervar1,kervar2);
       KY_test = KY_test./repmat(sqrt(sum(KY_test.^2)),[size(KY_test,1) 1]);  %//归一化
end
