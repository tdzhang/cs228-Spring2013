%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CliqueTreeCalibrate(P, isMax)


% Number of cliques in the tree.
N = length(P.cliqueList);

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j. 
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We have split the coding part for this function in two chunks with
% specific comments. This will make implementation much easier.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Now the clique tree has been calibrated. 
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




for i = 1:N
    for j = 1:N
        MESSAGES(i, j).val(1) = 0;
    end
end

if (~isMax)
    for iter=1:2*(N-1)
        [i, j] = GetNextCliques(P, MESSAGES);
        tmp = P.cliqueList(i);
        for k = 1:N
        if ((P.edges(k, i) ~= 0) && (k ~= j))
            tmp = FactorProduct(tmp, MESSAGES(k, i));
        end
        end
        MESSAGES(i, j) = FactorMarginalization(tmp, setdiff(P.cliqueList(i).var, intersect(P.cliqueList(i).var, P.cliqueList(j).var)));
        MESSAGES(i, j).val = MESSAGES(i, j).val / sum(MESSAGES(i, j).val);
    end
    for iter = 1:N
        tmp = P.cliqueList(iter);
        for s = 1:N
        if (P.edges(s, iter) ~= 0)
            tmp = FactorProduct(tmp, MESSAGES(s, iter));
        end
        end
        P.cliqueList(iter).val = tmp.val;
    end


else
    for iterate = 1:N
        P.cliqueList(iterate).val = log(P.cliqueList(iterate).val);
    end
    for iter=1:2*(N-1)
        [i, j] = GetNextCliques(P, MESSAGES);
        tmp = P.cliqueList(i);
        for k = 1:N
            if ((P.edges(k, i) ~= 0) && (k ~= j))
                tmp = FactorSum(tmp, MESSAGES(k, i));
            end
        end
        MESSAGES(i, j) = FactorMaxMarginalization(tmp, setdiff(P.cliqueList(i).var, intersect(P.cliqueList(i).var, P.cliqueList(j).var)));
    end
    for iter = 1:N
        tmp = P.cliqueList(iter);
        for s = 1:N
            if (P.edges(s, iter) ~= 0)
                tmp = FactorSum(tmp, MESSAGES(s, iter));
            end
        end
        P.cliqueList(iter).val = tmp.val;
    end
end


return
