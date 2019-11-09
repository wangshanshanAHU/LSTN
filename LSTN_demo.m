clear all
close all
addpath libsvm-new
addpath('liblinear-2.1/matlab');
addpath('util');
%%%%%%%%%
src_str = {'amazon','Caltech10','webcam','amazon','webcam','dslr','dslr','webcam','Caltech10','Caltech10','dslr','amazon'};
tgt_str = {'dslr','dslr','dslr','Caltech10','Caltech10','Caltech10','amazon','amazon','amazon','webcam','webcam','webcam'};

for i_sam = 1:length(tgt_str)
  
   src = src_str{i_sam};
    tgt = tgt_str{i_sam};
    fprintf(' %s vs %s ', src, tgt);
    
    load(['data\' src '_SURF_L10.mat']); 
    Xs = fts;
    Xs_label = labels;
    clear fts;
    clear labels;

    load(['data\' tgt '_SURF_L10.mat']); 
    Xt = fts;
    Xt_label = labels;
    clear fts;
    clear labels;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Xs_r = Xs./repmat(sqrt(sum(Xs.^2)),[size(Xs,1) 1]);  %//归一化
    Xt_r = Xt./repmat(sqrt(sum(Xt.^2)),[size(Xt,1) 1]);  %//列归一化 
    rate_ls=[];


    Xs_train=Xs_r';source_label_train= Xs_label;
    Xt_test=Xt_r';target_label_test=Xt_label;
    
    %%%%%%%%%Z_score归一化%%%%%%%%%%%%%%%%%%%%%
    Xs_train = zscore(Xs_train',1)';  %//归一化
    Xt_test = zscore(Xt_test',1)';  %//归一化
  %%%%%%%%%%L2norm归一化%%%%%%%%%%%%%%%%%
    Xs_train = Xs_train*diag(sparse(1./sqrt(sum(Xs_train.^2))));  %//归一化
    Xt_test = Xt_test*diag(sparse(1./sqrt(sum(Xt_test.^2))));  %//归一化
    
    ker_type=1; % 0:linear 1:nonlinear
   if ker_type==0
      Kernel='linear'; 
   else
     Kernel='gauss'; 
   end
   
   [Xs_train, Xt_test] = KN_function(Kernel,Xs_train,Xt_test);
   source_label_train_ori=source_label_train;
   target_label_test_ori=target_label_test;
   Xs_train_ori=Xs_train;
    Xt_test_ori=Xt_test;
     Ns=size(Xs_train,2);
     Nt=size(Xt_test,2);

    % ------------------------------------------
    %             Transfer Learning
    % ------------------------------------------ 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
  %one hot label
  for iter=1:3
  Y_source_label_train=one_hot_label(source_label_train);
  [opts] = PRES_parameter_setting(Xs_train); 
  [X_classify,net_PRES,Xs_new,Xt_test_new] = Lowranknorm_of_PRES_DTML_train(Xs_train,opts,Xt_test);
  end
  
rate = PRES_DTML_test(Xs_new,source_label_train_ori,Xt_test_new,target_label_test_ori,opts);
rate_ls(iter)=rate;
fprintf('rate_ls(%d)=%2.2f%%\n',iter,rate_ls(iter)); 

end   
    
    
    