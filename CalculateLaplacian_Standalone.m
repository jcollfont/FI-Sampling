%Calculate the Laplacian as a standalone because computationally intensive
load('10_2_17_test_rev1.mat')

%% Calculate Graph Laplacian
sigma=1;
AdjacMat=zeros(size(flatImage,1),size(flatImage,1)); %Change to feature map!
for i=1:size(flatImage,1)
    for j=1:size(flatImage,1)
        AdjacMat(i,j)=exp((-1/(2*sigma^2))*norm(flatImage(i)-flatImage(j)));
    end
end

del=diag(sum(AdjacMat,1))-AdjacMat;
