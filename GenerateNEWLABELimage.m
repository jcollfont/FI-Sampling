function [seg] = GenerateNEWLABELimage()
A=[1 1 256 256; 1 1 256 256; 1 1 256 256; 256 256 256 256];
B=mat2gray(A);
imwrite(B,'NEWLABELtest.jpg');
seg=[1 1 2 2; 1 1 2 2; 1 1 2 2; 2 2 2 2];

% C1=zeros(100);
% x=1:1:100;
% norm = normpdf(x,50,10);
% norm=norm';
% C1=ones(100)*norm(end);
% for i=10:55
%     C1(i,:)=norm;
% end
% %C1=C1/sum(sum(C1));
% 
% C2=zeros(100);
% x=1:1:100;
% norm = normpdf(x,66,10);
% C2=ones(100)*norm(end);
% for i=10:40
%     C2(:,i)=norm;
% end
% %C2=C2/sum(sum(C2));
% 
% C=C1+C2;
% C=C/sum(sum(C));
% 
% % figure()
% % imagesc(C1)
% 
% % figure()
% % imagesc(C2)
% 
% % figure()
% % imagesc(C)
% 
% PoolWeights=reshape(C1,100^2,1);
end