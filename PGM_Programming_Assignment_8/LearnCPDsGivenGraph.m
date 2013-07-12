function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

loglikelihood = 0;
P.c = zeros(1,K);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:K
    P.c(i) = sum(labels(:,i))/sum(sum(labels));
end

if (length(size(G)) > 2)
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
else
    for i = 1:size(G,1)
        if (G(i,1) == 0)
            for j = 1:K  %use every column of labels to classify dataset into K classes
                [P.clg(i).mu_y(j) P.clg(i).sigma_y(j)] = FitGaussianParameters(dataset(find(labels(:,j) == 1),i,1));
                [P.clg(i).mu_x(j) P.clg(i).sigma_x(j)] = FitGaussianParameters(dataset(find(labels(:,j) == 1),i,2));
                [P.clg(i).mu_angle(j) P.clg(i).sigma_angle(j)] = FitGaussianParameters(dataset(find(labels(:,j) == 1),i,3));
            end
        else
            P.clg(i).mu_y = [];
            P.clg(i).mu_x = [];
            P.clg(i).mu_angle = [];
            for j = 1:K
                [theta_y P.clg(i).sigma_y(j)] = FitLinearGaussianParameters(dataset(find(labels(:,j) == 1),i,1), reshape(dataset(find(labels(:,j) == 1),G(i,2),:),length(find(labels(:,j) == 1)),size(dataset,3)));
                [theta_x P.clg(i).sigma_x(j)] = FitLinearGaussianParameters(dataset(find(labels(:,j) == 1),i,2), reshape(dataset(find(labels(:,j) == 1),G(i,2),:),length(find(labels(:,j) == 1)),size(dataset,3)));
                [theta_angle P.clg(i).sigma_angle(j)] = FitLinearGaussianParameters(dataset(find(labels(:,j) == 1),i,3), reshape(dataset(find(labels(:,j) == 1),G(i,2),:),length(find(labels(:,j) == 1)),size(dataset,3)));
                P.clg(i).theta(j,:) = [theta_y(end),theta_y(1:end-1)',theta_x(end),theta_x(1:end-1)',theta_angle(end),theta_angle(1:end-1)'];
            end
        end
    end	
end

loglikelihood = ComputeLogLikelihood(P, G, dataset);

fprintf('log likelihood: %f\n', loglikelihood);

