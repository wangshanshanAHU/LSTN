function [Xt_test_update,Xt_label_update,Xt_test_rest,target_label_test_rest] = Ylabel_pilot_learning(Xt_test,Xt_test_new,target_label_test,opts,pilot_number,Xs_train,source_label_train)

%   [~,X_classify] =  PRESDTML_forward(Xt_test,W,B,opts);
%         if strcmp(opts.distType,'softmax')  %'euclidean'     
%            X_classify = exp(bsxfun(@minus, X_classify, max(X_classify,[],2)));
%            X_classify = bsxfun(@rdivide, X_classify, sum(X_classify, 2)); 
%         end         
        
    % train SVM on source training data
    Xs_train=Xs_train';
    Xt_test_new=Xt_test_new';
    model = train(source_label_train, sparse(double(Xs_train)), '-s 3 -c 1 -B 1 -q');

% predict SVM on target test data
[Y, ~, prob_estimates] = predict(target_label_test, sparse(double(Xt_test_new)), model, '-q'); 
 d2=diff([Y';target_label_test']);
 N2 = numel(find(d2==0));
accur_test=N2/size(target_label_test,1)*100;
fprintf('accur_test=%2.2f%%\n',accur_test); 
score=max(prob_estimates,[],2);
        
%        X=X_classify';     
%    for i=1:size(X,1)
%        [v,p]=max(X(i,:));
%        Y(i)=p;
%        score(i)=v;
%    end  
%        Y=Y';  

   b=sort(score,'descend');
   c=b(pilot_number);
   idx=find(score>=c);
   idx_rest=find(score<c);
 
%    idx=find(Y==target_label_test);
   Xt_test_update=Xt_test(:,idx);
   Xt_test_rest= Xt_test(:,idx_rest);
   target_label_test_rest=target_label_test(idx_rest);
   Xt_label_update=Y(idx);
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
