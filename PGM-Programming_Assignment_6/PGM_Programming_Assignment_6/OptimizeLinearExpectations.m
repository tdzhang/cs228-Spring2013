% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeLinearExpectations( I )
  % Inputs: An influence diagram I with a single decision node and one or more utility nodes.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  % You may assume that there is a unique optimal decision.
  %
  % This is similar to OptimizeMEU except that we will have to account for
  % multiple utility factors.  We will do this by calculating the expected
  % utility factors and combining them, then optimizing with respect to that
  % combined expected utility factor.  
  MEU = [];
  OptimalDecisionRule = [];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  %
  % A decision rule for D assigns, for each joint assignment to D's parents, 
  % probability 1 to the best option from the EUF for that joint assignment 
  % to D's parents, and 0 otherwise.  Note that when D has no parents, it is
  % a degenerate case we can handle separately for convenience.
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
F = [I.RandomFactors I.UtilityFactors(1)];
combine = VariableElimination(F, setdiff(unique([F(:).var]), I.DecisionFactors.var));
tempVariable = combine(1);
for j = 2:length(combine)
    tempVariable = FactorProduct(tempVariable, combine(j));
end
combine = tempVariable;

for i = 2:length(I.UtilityFactors)
    F = [I.RandomFactors I.UtilityFactors(i)];
    EUF = VariableElimination(F, setdiff(unique([F(:).var]), I.DecisionFactors.var));
    
    tempVariable = EUF(1);
    for j = 2:length(EUF)
        tempVariable = FactorProduct(tempVariable, EUF(j));
    end
    EUF = tempVariable;
    
    combine = FactorSum(combine, EUF);
end
EUF = combine;

D = I.DecisionFactors(1);

s = 0;
for i = 1:2:length(D.val)
    if (EUF.val(i) >= EUF.val(i+1))
        s = s + EUF.val(i);
        index(i) = 1;
        index(i+1) = 0;
    else
        s = s + EUF.val(i+1);
        index(i) = 0;
        index(i+1) = 1;
    end
end

MEU = s;
D.val = index;
OptimalDecisionRule = D;
end
