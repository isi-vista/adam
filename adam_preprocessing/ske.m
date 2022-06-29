% copied from https://github.com/ASU-APG/adam-stage/tree/main/processing
% original code by Sheng Cheng
function [S, E] = ske(path)
img = imread(path);
gray_img = rgb2gray(img);
bw = edge(gray_img, 'Canny');
dilate = imdilate(bw,strel('square', 5));
T = bwmorph(dilate,'thin',inf);
J1 = extract_junctions(T);
J = imbothat(dilate, strel('disk', 5));
P = dilate;
J = imdilate(J, strel('disk', 4)) & T;
J = J | J1;
I = imdilate(T, strel('square', 5));
U = trace_graph(T,J,I,P);

S = U.S;
E = U.link_ei_to_ni;

end


