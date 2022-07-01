% some modifications by Sheng Cheng
%
% Compute a bottom-up character skeleton.
% This algorithm should be deterministic.
%
% Input
%  I: [N x N] binary image (true = black)
%  bool_viz: visualize the results? (default = true)
%
%  Z: [struct] graph structure
%    fields
%    .n: number of nodes
%    .G: [n x 2] node coordinates
%    .E: [n x n boolean] adjacency matrix
%    .EI: [n x n cell], each cell is that edge's index into trajectory list
%         S. It should be a cell array, because there could be two paths
%    .S: [k x 1 cell] edge paths in the image 
%
function [U, T, J, P] = extract_skeleton(I)
    
    assert(UtilImage.check_black_is_true(I));
    
    P = logical(round(I/255.0));
    T = bwmorph(P,'thin',inf);
%     T = logical(round(T/255.0));
    
    %T = make_thin(I); % get thinned image
    J = extract_junctions(T); % get endpoint/junction features of thinned image
    U = trace_graph(T,J,I,P); % trace paths between features
    B = U.copy();
    U.clean_skeleton;
    
end

% Apply thinning algorithm. First it closes holes
% in the image.
%
% Input
%  I: [n x n boolean] raw image.
%    images are binary, where true means "black"
%
% Output
%  T: [n x n boolean] thinned image.
function T = make_thin(I)
    %I = bwmorph(I,'fill');
    I = logical(round(I/255.0));
    T = bwmorph(I,'thin',inf);
end