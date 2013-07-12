function [P G loglikelihood] = LearnGraphAndCPDs(dataset, labels)

% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha) 
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

G = zeros(10,2,K); % graph structures to learn
% initialization
for k=1:K
    G(2:end,:,k) = ones(9,2);
end

% estimate graph structure for each class
for k=1:K
    % fill in G(:,:,k)
    % use ConvertAtoG to convert a maximum spanning tree to a graph G
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%
    [A W] = LearnGraphStructure(dataset(find(labels(:,k) == 1),:,:));
    G(:,:,k) = ConvertAtoG(A);
end

% estimate parameters

P.c = zeros(1,K);
% compute P.c

% the following code can be copied from LearnCPDsGivenGraph.m
% with little or no modification
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P.c = sum(labels)/N;
P.clg = repmat(struct('mu_y',[],'sigma_y',[],'mu_x',[],'sigma_x',[],'mu_angle',[],'sigma_angle',[],'theta',[]),1,10);
for i=1:size(dataset,2)
    Edf = 0;
    if(G(i,1,1)==1)
        Edf = G(i,2,1);
    end
    if(Edf==0)
        [P.clg(i).mu_y(1) P.clg(i).sigma_y(1)] = FitGaussianParameters(dataset(find(labels(:,1)==1),i,1));
        [P.clg(i).mu_x(1) P.clg(i).sigma_x(1)] = FitGaussianParameters(dataset(find(labels(:,1)==1),i,2));
        [P.clg(i).mu_angle(1) P.clg(i).sigma_angle(1)] = FitGaussianParameters(dataset(find(labels(:,1)==1),i,3));
    else
        [Beta sigma] = FitLinearGaussianParameters(dataset(find(labels(:,1)==1),i,1), reshape(dataset(find(labels(:,1)==1),Edf,:),length(find(labels(:,1)==1)),3));
        P.clg(i).theta(1,2:4)=Beta(1:3);
        P.clg(i).theta(1,1)=Beta(4);
        P.clg(i).sigma_y(1) = sigma;
        [Beta sigma] = FitLinearGaussianParameters(dataset(find(labels(:,1)==1),i,2), reshape(dataset(find(labels(:,1)==1),Edf,:),length(find(labels(:,1)==1)),3));
        P.clg(i).theta(1,6:8)=Beta(1:3);
        P.clg(i).theta(1,5)=Beta(4);
        P.clg(i).sigma_x(1) = sigma;
        [Beta sigma] = FitLinearGaussianParameters(dataset(find(labels(:,1)==1),i,3), reshape(dataset(find(labels(:,1)==1),Edf,:),length(find(labels(:,1)==1)),3));
        P.clg(i).theta(1,10:12)=Beta(1:3);
        P.clg(i).theta(1,9)=Beta(4);
        P.clg(i).sigma_angle(1) = sigma;
    end
    Edf=0;
    if(G(i,1,2)==1)
        Edf =G(i,2,2);
    end
    if(Edf==0)
        [P.clg(i).mu_angle(2) P.clg(i).sigma_angle(2)] = FitGaussianParameters(dataset(find(labels(:,2)==1),i,3));
        [P.clg(i).mu_y(2) P.clg(i).sigma_y(2)] = FitGaussianParameters(dataset(find(labels(:,2)==1),i,1));
        [P.clg(i).mu_x(2) P.clg(i).sigma_x(2)] = FitGaussianParameters(dataset(find(labels(:,2)==1),i,2));
    else
        [Beta sigma] = FitLinearGaussianParameters(dataset(find(labels(:,2)==1),i,1), reshape(dataset(find(labels(:,2)==1),Edf,:),length(find(labels(:,2)==1)),3));
        P.clg(i).sigma_y(2) = sigma;
        P.clg(i).theta(2,2:4)=Beta(1:3);
        P.clg(i).theta(2,1)=Beta(4);
        [Beta sigma] = FitLinearGaussianParameters(dataset(find(labels(:,2)==1),i,2), reshape(dataset(find(labels(:,2)==1),Edf,:),length(find(labels(:,2)==1)),3));
        P.clg(i).sigma_x(2) = sigma;
        P.clg(i).theta(2,6:8)=Beta(1:3);
        P.clg(i).theta(2,5)=Beta(4);
        [Beta sigma] = FitLinearGaussianParameters(dataset(find(labels(:,2)==1),i,3), reshape(dataset(find(labels(:,2)==1),Edf,:),length(find(labels(:,2)==1)),3));
        P.clg(i).sigma_angle(2) = sigma;
        P.clg(i).theta(2,10:12)=Beta(1:3);
        P.clg(i).theta(2,9)=Beta(4);
    end
end
loglikelihood = ComputeLogLikelihood(P, G, dataset);
fprintf('log likelihood: %f\n', loglikelihood);