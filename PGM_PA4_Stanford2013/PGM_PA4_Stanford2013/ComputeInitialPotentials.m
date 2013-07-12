%COMPUTEINITIALPOTENTIALS Sets up the cliques in the clique tree that is
%passed in as a parameter.
%
%   P = COMPUTEINITIALPOTENTIALS(C) Takes the clique tree skeleton C which is a
%   struct with three fields:
%   - nodes: cell array representing the cliques in the tree.
%   - edges: represents the adjacency matrix of the tree.
%   - factorList: represents the list of factors that were used to build
%   the tree. 
%   
%   It returns the standard form of a clique tree P that we will use through 
%   the rest of the assigment. P is struct with two fields:
%   - cliqueList: represents an array of cliques with appropriate factors 
%   from factorList assigned to each clique. Where the .val of each clique
%   is initialized to the initial potential of that clique.
%   - edges: represents the adjacency matrix of the tree. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function P = ComputeInitialPotentials(C)

% number of cliques
N = length(C.nodes);

% initialize cluster potentials 
P.cliqueList = repmat(struct('var', [], 'card', [], 'val', []), N, 1);
P.edges = zeros(N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% First, compute an assignment of factors from factorList to cliques. 
% Then use that assignment to initialize the cliques in cliqueList to 
% their initial potentials. 

% C.nodes is a list of cliques.
% So in your code, you should start with: P.cliqueList(i).var = C.nodes{i};
% Print out C to get a better understanding of its structure.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P.edges=C.edges;
M= length(C.factorList);
flag=ones(1,M);

max=0;
for i=1:M
    for j=1:length(C.factorList(i).var)
       if C.factorList(i).var(j)>max
           max=C.factorList(i).var(j);
       end
    end
end

Card_Map=zeros(1,max);

for i=1:M
    for j = 1:length(C.factorList(i).var)
        Card_Map(C.factorList(i).var(j))=C.factorList(i).card(j);
    end
end

for i=1:N
    P.cliqueList(i).var=C.nodes{i};
    card=[];
    val_num=1;
    for k1=1:length(C.nodes{i})
        card=[card Card_Map(C.nodes{i}(k1))];
        val_num=val_num*Card_Map(C.nodes{i}(k1));
    end
    P.cliqueList(i).card=card;
    P.cliqueList(i).val=ones(1,val_num);

   for j=1:M
      if(isequal(intersect(C.nodes{i},C.factorList(j).var),intersect(C.factorList(j).var,C.factorList(j).var))&&flag(j))

              P.cliqueList(i) = FactorProduct(P.cliqueList(i), C.factorList(j));
              flag(j)=0;

      end
   end
end


end

