function [] = HeatPlots_11_26_17(Estimate_Matrix, iteration, NewLabels, UnlabeledIndices, trA, im,Fit)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure()
colormap('gray')
imagesc(Estimate_Matrix)
colorbar
title(['Iteration =',num2str(iteration)])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot heatmap
trA_new=zeros(length(NewLabels),1);
for i=1:length(UnlabeledIndices)
    trA_new(UnlabeledIndices(i))=trA(i);
end

%A_heat=zeros(size(im,2),size(im,1));
A_heat=reshape(trA_new,size(im,2),size(im,1));
% for i=1:length(coordinates)
%     A_heat(coordinates(i,2),coordinates(i,1))=trA_new(i);
% end

%if mod(iteration,20)==0 %iteration==IterationNum
    figure()
    colormap('hot')
    imagesc(A_heat)
    colorbar
%end

KernelSize=sqrt(size(Fit.w,1)-1);
figure()
imagesc(reshape(Fit.w(1:end-1,1),[KernelSize,KernelSize]))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%