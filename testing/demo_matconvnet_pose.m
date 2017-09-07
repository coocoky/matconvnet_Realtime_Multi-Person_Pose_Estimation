clear; close all; clc;
addpath('src');
addpath('../matconvnet/matlab');vl_setupnn;
mode = 1;vis = 1;
param = config(mode);
netStruct = load('../models/matconvnet_coco_openpose_model.mat') ;
net = dagnn.DagNN.loadobj(netStruct) ;
net.move('gpu');net.mode = 'test';
clear netStruct ;

close all;
oriImg = imread('./sample_image/ski.jpg');
scale0 = 368/size(oriImg, 1);
twoLevel = 1;
[final_score, ~] = applyModel(oriImg, param, net, scale0, 1, 1, 0, twoLevel);
if mode == 1
    [candidates, subset] = connect56LineVec(oriImg, final_score, param, vis);
elseif mode == 2
    [candidates, subset] = connect43LineVec(oriImg, final_score, param, vis);
end
pause;