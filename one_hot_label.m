function Y=one_hot_label(label)
   Class= length(unique(label));
   Class= 10;
   Yat= label;%target_label_train
   Y=0*ones(length(Yat),Class); 
   for j=1:length(Yat)   
       Y(j,Yat(j))=1; 
   end
   Y=Y';
end