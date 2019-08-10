load('Output_ogMimL6_NN_12_12_17.mat')

for a=1:length(Output)
    figure()
    hold on
    for b=2:2:length(Output(a).PoolIt)
        AT=Output(a).PoolIt(b).AccuracyTotal;
        plot(AT,'b')
    end
    for b=1:2:length(Output(a).PoolIt)
        AT=Output(a).PoolIt(b).AccuracyTotal;
        plot(AT)
    end
    title(['Total Accuracy for lambda=' num2str(Output(a).Lambda) ' and lambdaI=' num2str(Output(a).lambdaEye)])
    xlabel('Sampling Iteration')
    ylabel('Accuracy (0 to 1)')
    legend('Random 1', 'Run 1' , 'Run 2', 'Run 3') %'Random 2', 'Random 3',
    hold off;
end

% for a=1:length(Output)
%     figure()
%     hold on
%     for b=1:length(Output(a).PoolIt)
%         AT=Output(a).PoolIt(b).AccuracyTotal;
%         plot(AT)
%     end
%     title(['Total Accuracy for lambda=' num2str(Output(a).Lambda) ' and lambdaI=' num2str(Output(a).lambdaEye)])
%     xlabel('Sampling Iteration')
%     ylabel('Accuracy (0 to 1)')
%     legend('Random','Run 1', 'Run 2', 'Run 3')
%     hold off;
% end
    
% for a=1:length(Output)
%     figure()
%     hold on
%     for b=1:1
%         ATr=Output(a).PoolIt(b).AccuracyTotal;
%         plot(ATr)
%     end
%     AT=zeros(length(ATr));
%     for b=2:length(Output(a).PoolIt)
%         AT=AT+Output(a).PoolIt(b).AccuracyTotal;
%     end
%     AT=AT./3;
%     plot(AT)
%     title(['Total Accuracy for lambda=' num2str(Output(a).Lambda) ' and lambdaI=' num2str(Output(a).lambdaEye)])
%     xlabel('Sampling Iteration')
%     ylabel('Accuracy (0 to 1)')
%     legend('Random','Average of Runs')
%     hold off;
% end

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
