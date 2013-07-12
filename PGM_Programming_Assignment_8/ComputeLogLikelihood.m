function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of examples
K = length(P.c); % number of classes

loglikelihood = 0;
% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logy1 = zeros(N,10);
logx1 = zeros(N,10);
logalpha1 = zeros(N,10);
logy2 = zeros(N,10);
logx2 = zeros(N,10);
logalpha2 = zeros(N,10);

if (length(size(G)) > 2)
    for i = 1:size(G,1)
        if (G(i,1,1) == 0)  
            logy1(:,i) = lognormpdf(dataset(:,i,1), P.clg(i).mu_y(1), P.clg(i).sigma_y(1));  
            logx1(:,i) = lognormpdf(dataset(:,i,2), P.clg(i).mu_x(1), P.clg(i).sigma_x(1)); 
            logalpha1(:,i) = lognormpdf(dataset(:,i,3), P.clg(i).mu_angle(1), P.clg(i).sigma_angle(1));
        end
        if (G(i,1,2) == 0)
            logy2(:,i) = lognormpdf(dataset(:,i,1), P.clg(i).mu_y(2), P.clg(i).sigma_y(2));   
            logx2(:,i) = lognormpdf(dataset(:,i,2), P.clg(i).mu_x(2), P.clg(i).sigma_x(2)); 
            logalpha2(:,i) = lognormpdf(dataset(:,i,3), P.clg(i).mu_angle(2), P.clg(i).sigma_angle(2));
        end
        if (G(i,1,1) == 1)
            for j = 1:N    
                logy1(j,i) = lognormpdf(dataset(j,i,1), P.clg(i).theta(1,1) + P.clg(i).theta(1,2)*dataset(j,G(i,2,1),1) + P.clg(i).theta(1,3)*dataset(j,G(i,2,1),2) + P.clg(i).theta(1,4)*dataset(j,G(i,2,1),3), P.clg(i).sigma_y(1));   
                logx1(j,i) = lognormpdf(dataset(j,i,2), P.clg(i).theta(1,5) + P.clg(i).theta(1,6)*dataset(j,G(i,2,1),1) + P.clg(i).theta(1,7)*dataset(j,G(i,2,1),2) + P.clg(i).theta(1,8)*dataset(j,G(i,2,1),3), P.clg(i).sigma_x(1));
                logalpha1(j,i) = lognormpdf(dataset(j,i,3), P.clg(i).theta(1,9) + P.clg(i).theta(1,10)*dataset(j,G(i,2,1),1) + P.clg(i).theta(1,11)*dataset(j,G(i,2,1),2) + P.clg(i).theta(1,12)*dataset(j,G(i,2,1),3), P.clg(i).sigma_angle(1));
            end
        end
        if (G(i,1,2) == 1)
            for j = 1:N 
                logy2(j,i) = lognormpdf(dataset(j,i,1), P.clg(i).theta(2,1) + P.clg(i).theta(2,2)*dataset(j,G(i,2,2),1) + P.clg(i).theta(2,3)*dataset(j,G(i,2,2),2) + P.clg(i).theta(2,4)*dataset(j,G(i,2,2),3), P.clg(i).sigma_y(2));   
                logx2(j,i) = lognormpdf(dataset(j,i,2), P.clg(i).theta(2,5) + P.clg(i).theta(2,6)*dataset(j,G(i,2,2),1) + P.clg(i).theta(2,7)*dataset(j,G(i,2,2),2) + P.clg(i).theta(2,8)*dataset(j,G(i,2,2),3), P.clg(i).sigma_x(2));
                logalpha2(j,i) = lognormpdf(dataset(j,i,3), P.clg(i).theta(2,9) + P.clg(i).theta(2,10)*dataset(j,G(i,2,2),1) + P.clg(i).theta(2,11)*dataset(j,G(i,2,2),2) + P.clg(i).theta(2,12)*dataset(j,G(i,2,2),3), P.clg(i).sigma_angle(2));
            end
        end
    end
else  
    for i = 1:size(G,1)
        if (G(i,1) == 0)  
            logy1(:,i) = lognormpdf(dataset(:,i,1), P.clg(i).mu_y(1), P.clg(i).sigma_y(1));   
            logx1(:,i) = lognormpdf(dataset(:,i,2), P.clg(i).mu_x(1), P.clg(i).sigma_x(1)); 
            logalpha1(:,i) = lognormpdf(dataset(:,i,3), P.clg(i).mu_angle(1), P.clg(i).sigma_angle(1));
            logy2(:,i) = lognormpdf(dataset(:,i,1), P.clg(i).mu_y(2), P.clg(i).sigma_y(2));   
            logx2(:,i) = lognormpdf(dataset(:,i,2), P.clg(i).mu_x(2), P.clg(i).sigma_x(2)); 
            logalpha2(:,i) = lognormpdf(dataset(:,i,3), P.clg(i).mu_angle(2), P.clg(i).sigma_angle(2));
        else
            for j = 1:N    
                logy1(j,i) = lognormpdf(dataset(j,i,1), P.clg(i).theta(1,1) + P.clg(i).theta(1,2)*dataset(j,G(i,2),1) + P.clg(i).theta(1,3)*dataset(j,G(i,2),2) + P.clg(i).theta(1,4)*dataset(j,G(i,2),3), P.clg(i).sigma_y(1));   
                logx1(j,i) = lognormpdf(dataset(j,i,2), P.clg(i).theta(1,5) + P.clg(i).theta(1,6)*dataset(j,G(i,2),1) + P.clg(i).theta(1,7)*dataset(j,G(i,2),2) + P.clg(i).theta(1,8)*dataset(j,G(i,2),3), P.clg(i).sigma_x(1));
                logalpha1(j,i) = lognormpdf(dataset(j,i,3), P.clg(i).theta(1,9) + P.clg(i).theta(1,10)*dataset(j,G(i,2),1) + P.clg(i).theta(1,11)*dataset(j,G(i,2),2) + P.clg(i).theta(1,12)*dataset(j,G(i,2),3), P.clg(i).sigma_angle(1));
                logy2(j,i) = lognormpdf(dataset(j,i,1), P.clg(i).theta(2,1) + P.clg(i).theta(2,2)*dataset(j,G(i,2),1) + P.clg(i).theta(2,3)*dataset(j,G(i,2),2) + P.clg(i).theta(2,4)*dataset(j,G(i,2),3), P.clg(i).sigma_y(2));   
                logx2(j,i) = lognormpdf(dataset(j,i,2), P.clg(i).theta(2,5) + P.clg(i).theta(2,6)*dataset(j,G(i,2),1) + P.clg(i).theta(2,7)*dataset(j,G(i,2),2) + P.clg(i).theta(2,8)*dataset(j,G(i,2),3), P.clg(i).sigma_x(2));
                logalpha2(j,i) = lognormpdf(dataset(j,i,3), P.clg(i).theta(2,9) + P.clg(i).theta(2,10)*dataset(j,G(i,2),1) + P.clg(i).theta(2,11)*dataset(j,G(i,2),2) + P.clg(i).theta(2,12)*dataset(j,G(i,2),3), P.clg(i).sigma_angle(2));
            end
        end
    end
end
logL = zeros(1,N);
for j = 1:N
	logL(j) = log(exp(sum(logy1(j,:)) + sum(logx1(j,:)) + sum(logalpha1(j,:)) + log(P.c(1))) + exp(sum(logy2(j,:)) + sum(logx2(j,:)) + sum(logalpha2(j,:)) + log(P.c(2))));
end
loglikelihood = sum(logL);