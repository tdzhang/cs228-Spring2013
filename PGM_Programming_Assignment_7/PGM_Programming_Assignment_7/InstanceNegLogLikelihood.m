% function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)
% returns the negative log-likelihood and its gradient, given a CRF with parameters theta,
% on data (X, y). 
%
% Inputs:
% X            Data.                           (numCharacters x numImageFeatures matrix)
%              X(:,1) is all ones, i.e., it encodes the intercept/bias term.
% y            Data labels.                    (numCharacters x 1 vector)
% theta        CRF weights/parameters.         (numParams x 1 vector)
%              These are shared among the various singleton / pairwise features.
% modelParams  Struct with three fields:
%   .numHiddenStates     in our case, set to 26 (26 possible characters)
%   .numObservedStates   in our case, set to 2  (each pixel is either on or off)
%   .lambda              the regularization parameter lambda
%
% Outputs:
% nll          Negative log-likelihood of the data.    (scalar)
% grad         Gradient of nll with respect to theta   (numParams x 1 vector)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)

    % featureSet is a struct with two fields:
    %    .numParams - the number of parameters in the CRF (this is not numImageFeatures
    %                 nor numFeatures, because of parameter sharing)
    %    .features  - an array comprising the features in the CRF.
    %
    % Each feature is a binary indicator variable, represented by a struct 
    % with three fields:
    %    .var          - a vector containing the variables in the scope of this feature
    %    .assignment   - the assignment that this indicator variable corresponds to
    %    .paramIdx     - the index in theta that this feature corresponds to
    %
    % For example, if we have:
    %   
    %   feature = struct('var', [2 3], 'assignment', [5 6], 'paramIdx', 8);
    %
    % then feature is an indicator function over Y_2 and Y_3, which takes on a value of 1
    % if Y_2 = 5 and Y_3 = 6 (which would be 'e' and 'f'), and 0 otherwise. 
    % Its contribution to the log-likelihood would be theta(8) if it's 1, and 0 otherwise.
    %
    % If you're interested in the implementation details of CRFs, 
    % feel free to read through GenerateAllFeatures.m and the functions it calls!
    % For the purposes of this assignment, though, you don't
    % have to understand how this code works. (It's complicated.)
    
    featureSet = GenerateAllFeatures(X, modelParams);

    % Use the featureSet to calculate nll and grad.
    % This is the main part of the assignment, and it is very tricky - be careful!
    % You might want to code up your own numerical gradient checker to make sure
    % your answers are correct.
    %
    % Hint: you can use CliqueTreeCalibrate to calculate logZ effectively. 
    %       We have halfway-modified CliqueTreeCalibrate; complete our implementation 
    %       if you want to use it to compute logZ.
    
    nll = 0;
    grad = zeros(size(theta));
    %%%
    % Your code here:
    factorList = repmat(struct('var', [], 'card', [], 'val', []), length(featureSet.features), 1);
    for i = 1:length(featureSet.features)
        factorList(i).var = featureSet.features(i).var;
        factorList(i).card = modelParams.numHiddenStates * ones(1, length(factorList(i).var));
        factorList(i).val = ones(1, prod(factorList(i).card));
        I = AssignmentToIndex(featureSet.features(i).assignment, factorList(i).card);
        factorList(i).val(I) = exp(theta(featureSet.features(i).paramIdx));
    end
    P = CreateCliqueTree(factorList);
    [P, logZ] = CliqueTreeCalibrate(P, 0);
    
    weightFeatureCounts = 0;
    Edf = zeros(1, featureSet.numParams);
    eTheta = zeros(1, featureSet.numParams);
    for i = 1:length(featureSet.features)
        if (y(featureSet.features(i).var) == featureSet.features(i).assignment)
            weightFeatureCounts = weightFeatureCounts + theta(featureSet.features(i).paramIdx);
            Edf(featureSet.features(i).paramIdx) = Edf(featureSet.features(i).paramIdx) + 1;
        end
    end
    
    regulizationCost = modelParams.lambda / 2 * sum(theta.^2);
    nll = logZ - weightFeatureCounts + regulizationCost;
    for i = 1:length(featureSet.features)
        for j = 1:length(P.cliqueList)
            if intersect(featureSet.features(i).var, P.cliqueList(j).var) == featureSet.features(i).var
                f = FactorMarginalization(P.cliqueList(j), setdiff(P.cliqueList(j).var, featureSet.features(i).var));
                f.val = f.val / sum(f.val);
                eTheta(featureSet.features(i).paramIdx) = eTheta(featureSet.features(i).paramIdx) + GetValueOfAssignment(f, featureSet.features(i).assignment);
                break;
            end
        end
    end
    grad = eTheta - Edf + modelParams.lambda * theta;
end
