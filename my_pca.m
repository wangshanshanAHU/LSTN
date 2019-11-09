function [Pro_Matrix]=my_pca(Train_SET,Eigen_NUM)
%���룺
%Train_SET��ѵ����������ÿ����һ��������ÿ��һ��������Dim*Train_Num
%Eigen_NUM��ͶӰά��

%�����
%Pro_Matrix��ͶӰ����
%Mean_Image����ֵͼ��

[Dim,Train_Num]=size(Train_SET);

%��ѵ����������������ά��ʱ��ֱ�ӷֽ�Э�������
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
    %����С���󣬼���������ֵ������������Ȼ��ӳ�䵽�����
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
