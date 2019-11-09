function [rate_ls,Xt_test_update,Xt_label_update,Y] = Ylabel_PRES_DTML_test(Xt_test_ori,X_classify,W,B,net_PRES,Xt_test,target_label_test,opts)
% X: NxC
% Y: Nx1
%   [~,X_classify] =  PRESDTML_forward(Xt_test,W,B,opts);
        if strcmp(opts.distType,'softmax')  %'euclidean'     
           X_classify = exp(bsxfun(@minus, X_classify, max(X_classify,[],2)));
           X_classify = bsxfun(@rdivide, X_classify, sum(X_classify, 2)); 
        end  
        [rate4_1,T,Y]=decision_class(X_classify',target_label_test);    
        
   Y=Y';
   idx=find(Y==target_label_test);
   Xt_test_update=Xt_test_ori(:,idx);
   Xt_label_update=target_label_test(idx);
   rate_ls=rate4_1;
  end
             


  % -------------------------------------------
    %               Classification
    % -------------------------------------------
    % SRC  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%    Yat= Y_test;
%    Xat=[X_train(:,1:length(Xs_train_label_sample))];  % Xt_new
%   a=[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,100];
%    for j=1:length(a)
%    W=(Xat'*Xat+a(j)*eye(size(Xat,2)))\(Xat'*Yat);
%    [accuracy(i,j),xp,r]=computaccuracy(X_train,Class,Xs_train_label_sample',Y_test,target_label_test',W,a(j));      
%    end   
%    rate_ls(i)=max(accuracy(i,:));
%    fprintf('%2.2f%%\n',rate_ls(i));  
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
%   % svm
%     counter=0;
%   for m_svm=-5:5
%         c=10^m_svm;
%         for n_svm=-5:5
%             gama=10^n_svm;
%             counter=counter+1;
%             tmd=['-s 0 -t t -g ',num2str(c), ' -c ',num2str(gama)];
%             model = svmtrain(Xs_train_label_sample, X_train', tmd); 
%             [predict_label_test, accuracy_test] = svmpredict(target_label_test, Y_test', model);
%             d2=diff([predict_label_test';target_label_test']);
%             N2 = numel(find(d2==0));
%             accur_test=N2/size(target_label_test,1);
%            result(counter,1)=accuracy_test(1);   
%            result(counter,2)=accur_test;
% %            result(counter,3)=predict_label_test;
%    end
% end
% [rate_svm(i),index]=max(result(:,1));
%  acc = rate_svm(i);
%  acc_a(i)=rate_svm(i);
% fprintf('acc(%d)= %2.2f%%\n',i,acc); 
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%  end
%     ave_acc=mean(acc_a);
%     fprintf('ave_svm= %2.2f%%\n',ave_acc);
%     end