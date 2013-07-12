function M = ComputeMaxMarginal(V, F, E)

% Check for empty factor list
assert(numel(F) ~= 0, 'Error: empty factor list');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE:
% M should be a factor
% Remember to renormalize the entries of M!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  F = ObserveEvidence(F, E);
  Joint = ComputeJointDistribution(F);
 %Joint.val = Joint.val ./ sum(Joint.val);
  M = FactorMaxMarginalization(Joint, setdiff(Joint.var, V));
  %M.val = M.val ./ sum(M.val);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end