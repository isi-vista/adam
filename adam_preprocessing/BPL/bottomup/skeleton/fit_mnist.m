% Demo of fitting a motor program to an image.

% Parameters
K = 5; % number of unique parses we want to collect
verbose = false; % describe progress and visualize parse?
include_mcmc = false; % run mcmc to estimate local variability?
fast_mode = true; % skip the slow step of fitting strokes to details of the ink?

ps = defaultps;
load(ps.libname,'lib');

fid = fopen('t10k-images-idx3-ubyte', 'r');
fid2 = fopen('t10k-labels-idx1-ubyte', 'r');
A = fread(fid, 1, 'uint32');
magicnumber = swapbytes(uint32(A));
A = fread(fid, 1, 'uint32');
totalimages = swapbytes(uint32(A));
A = fread(fid, 1, 'uint32');
numrow = swapbytes(uint32(A));
A = fread(fid, 1, 'uint32');
numcol = swapbytes(uint32(A));
B = fread(fid2, 1, 'uint32');
swapbytes(uint32(B));
B = fread(fid2, 1, 'uint32');
swapbytes(uint32(B));
G = cell([8, totalimages]);
newscale = lib.newscale;
[ncat,dim] = size(lib.shape.mu);
ncpt = dim/2;
minlen = ncpt*2;
for k = 1: totalimages
    A = fread(fid, numrow*numcol, 'uint8');
    A = reshape(uint8(A), numrow, numcol)';
    B = fread(fid2, 1, 'uint8');
    U = extract_skeleton(A,false);
    S = U.S;
    [S_norm,S_offset,S_scales] = normalize_dataset(S,newscale,false);         
    nsub = length(S);
    S_splines = cell(nsub,1);
    for b=1:nsub
        traj = S_norm{b};        
        celltraj = expand_small_strokes({traj},minlen,true);
        traj = celltraj{1};
        bspline = fit_bspline_to_traj(traj,ncpt);
        S_splines{b} = bspline;
    end
    
%     G{1,k} = U;
    G{1,k} = U.I;
    G{2,k} = B;
    G{3,k} = S_splines;
    G{4,k} = {S_norm; S_offset; S_scales};
    G{5,k} = S;
    G{6,k} = U.G;
    G{7,k} = U.E;
    G{8,k} = U.link_ei_to_ni;
end
save('skeleton.mat', 'G');


% G = fit_motorprograms(img,K,verbose,include_mcmc,fast_mode);