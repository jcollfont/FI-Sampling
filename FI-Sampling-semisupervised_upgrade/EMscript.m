%Expectation Maximization Algorithm
clear;

%Inputs
%i. inputData: nSamples x dDimensions array with the data to be clustered
%ii. numberOfClusters: number of clusters for algorithm
%iii. stopTolerance: parameter for convergence criteria
%iv. maxIterations: maximum number of times to iterate before convergence
%is reached
%v. numberOfRuns: number of times the algo will run with random
%initializtions

load testDataset.mat
%load dataset1.mat

inputData = data;
numberOfClusters = 4;
stopTolerance = 10^-5;
maxIterations = 500;
numberOfRuns = 1;

dim=size(inputData,2); %number of dimensions
points=length(inputData);

%initialize means by taking random means
randIndex=randperm(length(inputData))'; %random indexing of data
meu(1:numberOfClusters,1:dim)=inputData(randIndex(1:numberOfClusters),:); %mean estimate (taken as random mean to start)

%initialize covariances as identity matrices
sigma=zeros(dim,dim,numberOfClusters);
for i=1:numberOfClusters
    sigma(:,:,i)=eye(dim);
end

%initialize priors as equal
priors(1:numberOfClusters,1)=(1/numberOfClusters)*ones(numberOfClusters,1); %weight of data with respect to classes

log_likelihood_old=0;
for iteration=1:maxIterations
     % E-Step
     respnum=zeros(numberOfClusters,points);
     for i=1:numberOfClusters
         for ii=1:points
             respnum(i,ii)=priors(i)*mvnpdf(inputData(ii,:),meu(i,:),sigma(:,:,i));
         end
    end
    resp=respnum/sum(sum(respnum));

    % M-step
    Nk=zeros(numberOfClusters,1);
    for i=1:numberOfClusters
        Nk(i)=sum(resp(i,:));
        meu(i,:)=(1/Nk(i))*((resp(i,:)*inputData));
        sigma(:,:,i)=(1/Nk(i))*sum((resp(i,:)))*(inputData-repmat(meu(i,:),points,1))'*(inputData-repmat(meu(i,:),points,1));
        %sigma(:,:,i)=(sigma(:,:,i) + sigma(:,:,i).') / 2;
        priors(i)=Nk(i);
    end
    priors=priors/sum(Nk);

    %Evaluate Log likelihood
    l=zeros(numberOfClusters,points);
    for i=1:numberOfClusters
        for ii=1:points
            l(i,ii)=log(priors(i)*mvnpdf(inputData(ii,:),meu(i,:),sigma(:,:,i)));
        end
    end
    log_likelihood(iteration)=sum(sum(l));
    if abs(log_likelihood_old-log_likelihood(iteration))<stopTolerance %if tolerance is within 10^-5
        break; %stop iterating
    end
    log_likelihood_old=log_likelihood(iteration);
end

%ii. estimatedLabels: nSamples x 1 vector with labels based on maximum
%probability. EM is a soft clustering algoirthm so output is just
%densities for each cluster in the mixture model

[m,c]=max(resp);
estimatedLabels=c-1;

figure()
gscatter(inputData(:,1),inputData(:,2),estimatedLabels)

%Outputs
%i. clusterParameters: numberOfClusters x 1 struct array with the Gaussian
%mixture parameters (.mu, .covariance, .prior)
%ii. estimatedLabels: nSamples x 1 vector with labels based on maximum
%probability. EM is a soft clustering algoirthm so output is just
%densities for each cluster in the mixture model
%iii. logLikelihood: 1 x numberOfIterations vector of loglikelihood as a
%function of iteration number
%iv. costVsComplexity: 1 x maxNumberOfClusters vector with BIC criteria as
%a function of number of clusters

% % log_likelihood_old=0; %start with zero log likelihood
% % p=zeros(1,length(data)); %row of zeros
% % probb=mvnpdf(inputData,meu1,sigma1); %probability of each point
% % p=(probb*weight); %weight*probabilities
% % 
% % 
% % for i = 1:maxIterations %for (up to) max iterations
% %     probb=mvnpdf(inputData,meu1,sigma1); %probability of each point
% %     p=(weight'*probb); %weight*probabilities
% %     log_likelihood(i)=sum(log(p)); %log likelihood is sum of log of weighted probabilites
% %     %M Step
% %     posterior=bsxfun(@rdivide,bsxfun(@times,weight,probb),p); %calculate posterior probabilites
% %     meu1=(posterior*inputData')./((sum(posterior'))'); %calculate new mean
% %     sigma1=(sum((posterior.*(bsxfun(@minus,inputData,meu1).^2))')./sum(posterior'))'; %calculate new covariance
% %     weight=(sum(posterior')./length(data))'; %reweight data points
% %     if abs(log_likelihood_old-log_likelihood(i))<stopTolerance %if tolerance is within 10^-5
% %         break; %stop iterating
% %     end
% %     log_likelihood_old=log_likelihood(i);
% % end
% % % 
% % % field1='mu';
% % % value1=meu1;
% % % field2='covariance';
% % % value2=sigma1;
% % % field3='prior';
% % % value3=probb;
% % % s = struct(field1,value1,field2,value2,field3,value3);
% % % 
% % % estimatedLabels=1; %?????
% % % 
% % % logLikelihood=log_likelihood;
% % % 
% % % costVsComplexity=1; %?????
% % 
