close all;
load('BWOutput_11_27_17.mat')


for a=1:length(BWOutput)
    figure()
    hold on
    for b=1:1
        AT=BWOutput(a).PoolIt(b).AccuracyTotal;
        plot(AT)
    end
    title(['Total Accuracy for lambda=' num2str(BWOutput(a).Lambda) ' and lambdaI=' num2str(BWOutput(a).lambdaEye)])
    xlabel('Sampling Iteration')
    ylabel('Accuracy (0 to 1)')
    hold off;
end
    
% figure()
% hold on
% for b=1:3
%     A1=BWOutput(a).PoolIt(b).Accuracy1;
%     plot(A1)
% end
% title(['Class 1 Accuracy for ' num2str(b) ' Pool Initializations'])
% xlabel('Sampling Iteration')
% ylabel('Accuracy (0 to 1)')
% hold off;
% 
% figure()
% hold on
% for b=1:3
%     A2=BWOutput(a).PoolIt(b).Accuracy2;
%     plot(A2)
% end
% title(['Class 2 Accuracy for ' num2str(b) ' Pool Initializations'])
% xlabel('Sampling Iteration')
% ylabel('Accuracy (0 to 1)')
% hold off;
% 
