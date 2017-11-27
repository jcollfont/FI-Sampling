A=ones(100)*256;
A(1:floor(end/2),1:floor(end/2))=1;
B=mat2gray(A);
imwrite(B,'BWtest.jpg');
seg=ones(100)*2;
seg(1:floor(end/2),1:floor(end/2))=1;
save('BWtestTruth.mat','seg','-v7.3');