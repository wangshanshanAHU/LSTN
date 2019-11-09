% addpath('data');
% load('trainingSet.mat','Xs','Ls');
% load('testingSet.mat','Xt','Lt');
% 
addpath('util');
% addpath('util/vlfeat-0.9.20');


Xs = rand(800,400);
Xt = randn(800,200);
Ls = randi(10,400,1);


%%%% conventional local binary patterns (LBP) gets 5900-dimensional feature vector %%%
%run('util/vlfeat-0.9.20/toolbox/vl_setup');
%vl_lbp();


%%%% PCA reduce it to 500 %%%%%

[coeff,score, latent] = princomp(Xs');
Xs = score(:,1:500)';
[coeff,score,latent] = princomp(Xt');
Xt = score(:,1:500)';

% input dimension is set to 500 %

opts.alpha = 0.1;
opts.beta = 10;
opts.gamma = 0.1;
opts.M = 2; % Layer number
opts.hidNum = [400,300];

opts.k1 = 5;
opts.k2 = 10;
opts.actfuncType = 'tanh';
opts.lr = 0.2; %%% initial learning rate
opts.lr_decay = 0.95;
opts.T = 10;
opts.distType = 'euclidean';
opts.epsilon = 1e-5;

net = DTML_train(Xs,Ls,Xt,opts);

DTML_test(net, Xt,Lt,opts);