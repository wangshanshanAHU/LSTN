function [opts] = PRES_parameter_setting(Xs_train)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
dimension=size(Xs_train,1);
opts.beta1 = 0.1;
opts.beta2 = 0.1;
opts.gamma2 =0.1;
opts.gamma1 = 0.01*0.1;
opts.M = 2; % Layer number
  opts.hidNum = [dimension,dimension];
%  opts.hidNum = [dimension,dimension*2,dimension*2,dimension*2,dimension];


opts.actfuncType = 'tanh';%sigmoid%tanh
opts.lr =  0.001; %%% initial learning rate
opts.lr_decay = 0.95;
opts.T =40;
opts.distType = 'euclidean';  %'softmax'euclidean
opts.epsilon = 1e-5;

end

