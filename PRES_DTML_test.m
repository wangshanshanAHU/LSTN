function rate_ls = RES_DTML_test(Xs_train,source_label_train,Xt_test,target_label_test,opts)
  Class= length(unique(source_label_train)); 
  % ls
  for i=1
  Yat=[source_label_train];%target_label_train
  Xat=[Xs_train(:,1:length(source_label_train))]';  % Xt_new
  Y=-1*ones(length(Yat),Class); 
   for j=1:length(Yat)   
       Y(j,Yat(j))=1; 
   end  
%    search the best regularization parameter
   a=[0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,100];
   for j=1:length(a)
       w=(Xat'*Xat+a(j)*eye(size(Xat,2)))\(Xat'*Y);
       yte1=Xt_test'*w;
       rate4_1(i,j)=decision_class(yte1,target_label_test); 
   end
   rate_ls=max(rate4_1(i,:));
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