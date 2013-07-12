% You should put all your code for recognizing unknown actions in this file.
% Describe the method you used in YourMethod.txt.
% Don't forget to call SavePrediction() at the end with your predicted labels to save them for submission, then submit using submit.m
function [ predicted_labels ] = RecognizeUnknownActions(G, maxIter)

load PA9Data;
datasetTrain = datasetTrain3;
datasetTest = datasetTest3;


% INPUTS
% datasetTrain: dataset for training models, see PA for details
% datasetTest: dataset for testing models, see PA for details
% G: graph parameterization as explained in PA decription
% maxIter: max number of iterations to run for EM

% OUTPUTS
% accuracy: recognition accuracy, defined as (#correctly classified examples / #total examples)
% predicted_labels: N x 1 vector with the predicted labels for each of the instances in datasetTest, with N being the number of unknown test instances


% Train a model for each action
% Note that all actions share the same graph parameterization and number of max iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ps = {};loglikelihood = {};ClassProbs = {};PairProbs = {};
for a=1:length(datasetTrain)
    InitialClassProb=datasetTrain(a).InitialClassProb;
    InitialClassProb=InitialClassProb+0.01;
    for i=1:size(InitialClassProb,1)
        InitialClassProb(i,:) = InitialClassProb(i,:) / sum(InitialClassProb(i,:));
    end
    InitialPairProb=datasetTrain(a).InitialPairProb;
    InitialPairProb=InitialPairProb+0.01;
    for i=1:size(InitialPairProb,1)
        InitialPairProb(i,:) = InitialPairProb(i,:) / sum(InitialPairProb(i,:));
    end
  [Ps{a} loglikelihood{a} ClassProb{a} PairProb{a}] = EM_HMM(datasetTrain(a).actionData, datasetTrain(a).poseData, G, InitialClassProb, InitialPairProb, maxIter);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Classify each of the instances in datasetTrain
% Compute and return the predicted labels and accuracy
% Accuracy is defined as (#correctly classified examples / #total examples)
% Note that all actions share the same graph parameterization

accuracy = 0;
predicted_labels = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ind_test=1:length(datasetTest.actionData)
  
  testllike = zeros(1, length(datasetTest));
  
  for action = 1:length(datasetTrain)
  
    poseData = datasetTest.poseData(datasetTest.actionData(ind_test).marg_ind, :, :);
    P = Ps{action};
    actionData = datasetTest.actionData(ind_test);
    
  
    N = length(actionData.marg_ind);
    K = size(ClassProb{action}, 2);
    logEmissionProb = zeros(N, K);

    for e=1:N
    for k=1:K
      for p=1:10
        parentpart = 0;
        parentals = [];
        if G(p, 1) == 1
          parentpart = G(p, 2);
          p_y = poseData(e, parentpart, 1);p_x = poseData(e, parentpart, 2);p_al = poseData(e, parentpart, 3);
          parentals = [ p_y p_x p_al ];
        end
        if (parentpart > 0)
          pdf_y = lognormpdf(poseData(e, p, 1), P.clg(p).theta(k, 1) + parentals * P.clg(p).theta(k, 2:4)', P.clg(p).sigma_y(k));
          pdf_x = lognormpdf(poseData(e, p, 2), P.clg(p).theta(k, 5) + parentals * P.clg(p).theta(k, 6:8)', P.clg(p).sigma_x(k));
          pdf_angle = lognormpdf(poseData(e, p, 3), P.clg(p).theta(k, 9) + parentals * P.clg(p).theta(k, 10:12)', P.clg(p).sigma_angle(k));
          logEmissionProb(e, k) = sum( [ logEmissionProb(e, k) pdf_y pdf_x pdf_angle ] );
        else
          pdf_x = lognormpdf(poseData(e, p, 2), P.clg(p).mu_x(k), P.clg(p).sigma_x(k));
          pdf_y = lognormpdf(poseData(e, p, 1), P.clg(p).mu_y(k), P.clg(p).sigma_y(k));
          pdf_angle = lognormpdf(poseData(e, p, 3), P.clg(p).mu_angle(k), P.clg(p).sigma_angle(k));
          logEmissionProb(e, k) = sum( [ logEmissionProb(e, k) pdf_y pdf_x pdf_angle ] );
        end
      end
    end
    end

    factorList = repmat(struct ('var', [], 'card', [], 'val', []), 1, 2 * N );
    cF = 1;

    factorList(cF).var = 1;
    factorList(cF).card = K;
    factorList(cF).val = log(P.c);
    cF = cF + 1;

    for i=2:N
      factorList(cF).var = [i-1 i];
      factorList(cF).card = [ K K ];
      factorList(cF).val = log(P.transMatrix(:)');
      cF = cF + 1;
    end
    

    for i = 1:N
      factorList(cF).var = i;
      factorList(cF).card = K;
      factorList(cF).val = logEmissionProb(i, :);
      cF = cF + 1;
    end
    
    [~,PCalibrated] = ComputeExactMarginalsHMM(factorList);
  
    testllike(action) = logsumexp(PCalibrated.cliqueList(end).val);
    
  end
  [ ~, predicted_labels(ind_test) ] = max(testllike); 
end

predicted_labels = predicted_labels';
SavePrediction(predicted_labels);