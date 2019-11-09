function [Xt_last_output,Xs_new,Xt_new] = computer_last_layer(Xt_test_ori,Xs_train_ori,W,B,net,opts)
   inputLayer_s = Xs_train_ori;
   inputLayer_test = Xt_test_ori;
for m = 1:opts.M %Öð²ãÇ°Ïò´«²¥
        [hs{m},zs{m}] =  PRESDTML_forward(inputLayer_s,net.layer{m}.W,net.layer{m}.b,opts);
        [ht_test{m},zt_test{m}] = PRESDTML_forward(inputLayer_test,net.layer{m}.W,net.layer{m}.b,opts);
        inputLayer_s = hs{m};
        inputLayer_test = ht_test{m};
end 
  Xs_new=hs{opts.M};
  Xt_new=ht_test{opts.M};
  [hs{opts.M+1},Xt_last_output] =  PRESDTML_forward(inputLayer_test,W,B,opts);
end