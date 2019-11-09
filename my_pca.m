function [Pro_Matrix]=my_pca(Train_SET,Eigen_NUM)
%输入：
%Train_SET：训练样本集，每列是一个样本，每行一类特征，Dim*Train_Num
%Eigen_NUM：投影维数

%输出：
%Pro_Matrix：投影矩阵
%Mean_Image：均值图像

[Dim,Train_Num]=size(Train_SET);

%当训练样本数大于样本维数时，直接分解协方差矩阵
if Dim<=Train_Num
    Mean_Image=mean(Train_SET,2);
    Train_SET=bsxfun(@minus,Train_SET,Mean_Image);
    R=Train_SET*Train_SET'/(Train_Num-1);
    
    [eig_vec,eig_val]=eig(R);
    eig_val=diag(eig_val);
    [~,ind]=sort(eig_val,'descend');
    W=eig_vec(:,ind);
    Pro_Matrix=W(:,1:Eigen_NUM);
    
else
    %构造小矩阵，计算其特征值和特征向量，然后映射到大矩阵
    Mean_Image=mean(Train_SET,2);
    Train_SET=bsxfun(@minus,Train_SET,Mean_Image);
    R=Train_SET'*Train_SET/(Train_Num-1);
    
    [eig_vec,eig_val]=eig(R);
    eig_val=diag(eig_val);
    [val,ind]=sort(eig_val,'descend');
    W=eig_vec(:,ind);
    Pro_Matrix=Train_SET*W(:,1:Eigen_NUM)*diag(val(1:Eigen_NUM).^(-1/2));
end

end
