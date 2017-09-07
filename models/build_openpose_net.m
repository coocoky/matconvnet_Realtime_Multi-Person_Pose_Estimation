function build_openpose_net()
addpath('../matconvnet/matlab')
vl_setupnn;
net = dagnn.DagNN() ;
lastAdded.var = 'image' ;
lastAdded.depth = 3 ;

opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
% -------------------------------------------------------------------------
% Add input section
% -------------------------------------------------------------------------

add_conv_relu('%s1_1', 3, 1, 64, 'relu', true) ;
add_conv_relu('%s1_2', 3, 1, 64, 'relu', true) ;
net.addLayer('pool1_stage1',...
    dagnn.Pooling('poolSize', [2 2], 'stride', 2, 'pad', 0,'method', 'max'), ...
    lastAdded.var, 'pool1_stage1');
lastAdded.var = 'pool1_stage1' ;

add_conv_relu('%s2_1', 3, 1, 128, 'relu', true) ;
add_conv_relu('%s2_2', 3, 1, 128, 'relu', true) ;
net.addLayer('pool2_stage1',...
    dagnn.Pooling('poolSize', [2 2], 'stride', 2, 'pad', 0,'method', 'max'), ...
    lastAdded.var, 'pool2_stage1');
lastAdded.var = 'pool2_stage1' ;

add_conv_relu('%s3_1', 3, 1, 256, 'relu', true) ;
add_conv_relu('%s3_2', 3, 1, 256, 'relu', true) ;
add_conv_relu('%s3_3', 3, 1, 256, 'relu', true) ;
add_conv_relu('%s3_4', 3, 1, 256, 'relu', true) ;
net.addLayer('pool3_stage1',...
    dagnn.Pooling('poolSize', [2 2], 'stride', 2, 'pad', 0,'method', 'max'), ...
    lastAdded.var, 'pool3_stage1');
lastAdded.var = 'pool3_stage1' ;

add_conv_relu('%s4_1', 3, 1, 512, 'relu', true) ;
add_conv_relu('%s4_2', 3, 1, 512, 'relu', true) ;
add_conv_relu('%s4_3_CPM', 3, 1, 256, 'relu', true) ;
add_conv_relu('%s4_4_CPM', 3, 1, 128, 'relu', true);

%% stage1
inputvarbase = lastAdded.var; %%important
inputdepthbase = lastAdded.depth;

lastAdded.var = inputvarbase;
lastAdded.depth = inputdepthbase;
add_conv_relu('%s5_1_CPM_L1', 3, 1, 128, 'relu', true) ;
add_conv_relu('%s5_2_CPM_L1', 3, 1, 128, 'relu', true) ;
add_conv_relu('%s5_3_CPM_L1', 3, 1, 128, 'relu', true) ;
add_conv_relu('%s5_4_CPM_L1', 1, 1, 512, 'relu', true) ;
add_conv_relu('%s5_5_CPM_L1', 1, 1, 38, 'relu', false) ;
outputvar1 = lastAdded.var;
outputdepth1 = lastAdded.depth;

lastAdded.var = inputvarbase;
lastAdded.depth = inputdepthbase;
add_conv_relu('%s5_1_CPM_L2', 3, 1, 128, 'relu', true) ;
add_conv_relu('%s5_2_CPM_L2', 3, 1, 128, 'relu', true) ;
add_conv_relu('%s5_3_CPM_L2', 3, 1, 128, 'relu', true) ;
add_conv_relu('%s5_4_CPM_L2', 1, 1, 512, 'relu', true) ;
add_conv_relu('%s5_5_CPM_L2', 1, 1, 19, 'relu', false) ;
outputvar2 = lastAdded.var;
outputdepth2 = lastAdded.depth;

net.addLayer('concat_stage2',...
    dagnn.Concat(), ...
    {outputvar1,outputvar2,inputvarbase}, 'concat_stage2');
lastAdded.var = 'concat_stage2' ;
lastAdded.depth = inputdepthbase + outputdepth1 + outputdepth2;

%% stage2
inputvarbase2 = lastAdded.var; %%important
inputdepthbase2 = lastAdded.depth;
lastAdded.var = inputvarbase2;
lastAdded.depth = inputdepthbase2;
add_conv_relu('M%s1_stage2_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s2_stage2_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s3_stage2_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s4_stage2_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s5_stage2_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s6_stage2_L1', 1, 1, 128, 'relu', true) ;
add_conv_relu('M%s7_stage2_L1', 1, 1, 38, 'relu', false) ;
outputvar1 = lastAdded.var;
outputdepth1 = lastAdded.depth;

lastAdded.var = inputvarbase2;
lastAdded.depth = inputdepthbase2;
add_conv_relu('M%s1_stage2_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s2_stage2_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s3_stage2_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s4_stage2_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s5_stage2_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s6_stage2_L2', 1, 1, 128, 'relu', true) ;
add_conv_relu('M%s7_stage2_L2', 1, 1, 19, 'relu', false) ;
outputvar2 = lastAdded.var;
outputdepth2 = lastAdded.depth;

net.addLayer('concat_stage3',...
    dagnn.Concat(), ...
    {outputvar1,outputvar2,inputvarbase}, 'concat_stage3');
lastAdded.var = 'concat_stage3' ;
lastAdded.depth = inputdepthbase + outputdepth1 + outputdepth2;


%% stage3
inputvarbase3 = lastAdded.var; %%important
inputdepthbase3 = lastAdded.depth;
lastAdded.var = inputvarbase3;
lastAdded.depth = inputdepthbase3;
add_conv_relu('M%s1_stage3_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s2_stage3_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s3_stage3_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s4_stage3_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s5_stage3_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s6_stage3_L1', 1, 1, 128, 'relu', true) ;
add_conv_relu('M%s7_stage3_L1', 1, 1, 38, 'relu', false) ;
outputvar1 = lastAdded.var;
outputdepth1 = lastAdded.depth;

lastAdded.var = inputvarbase3;
lastAdded.depth = inputdepthbase3;
add_conv_relu('M%s1_stage3_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s2_stage3_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s3_stage3_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s4_stage3_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s5_stage3_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s6_stage3_L2', 1, 1, 128, 'relu', true) ;
add_conv_relu('M%s7_stage3_L2', 1, 1, 19, 'relu', false) ;
outputvar2 = lastAdded.var;
outputdepth2 = lastAdded.depth;

net.addLayer('concat_stage4',...
    dagnn.Concat(), ...
    {outputvar1,outputvar2,inputvarbase}, 'concat_stage4');
lastAdded.var = 'concat_stage4' ;
lastAdded.depth = inputdepthbase + outputdepth1 + outputdepth2;

%% stage4
inputvarbase4 = lastAdded.var; %%important
inputdepthbase4 = lastAdded.depth;
lastAdded.var = inputvarbase4;
lastAdded.depth = inputdepthbase4;
add_conv_relu('M%s1_stage4_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s2_stage4_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s3_stage4_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s4_stage4_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s5_stage4_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s6_stage4_L1', 1, 1, 128, 'relu', true) ;
add_conv_relu('M%s7_stage4_L1', 1, 1, 38, 'relu', false) ;
outputvar1 = lastAdded.var;
outputdepth1 = lastAdded.depth;

lastAdded.var = inputvarbase4;
lastAdded.depth = inputdepthbase4;
add_conv_relu('M%s1_stage4_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s2_stage4_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s3_stage4_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s4_stage4_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s5_stage4_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s6_stage4_L2', 1, 1, 128, 'relu', true) ;
add_conv_relu('M%s7_stage4_L2', 1, 1, 19, 'relu', false) ;
outputvar2 = lastAdded.var;
outputdepth2 = lastAdded.depth;

net.addLayer('concat_stage5',...
    dagnn.Concat(), ...
    {outputvar1,outputvar2,inputvarbase}, 'concat_stage5');
lastAdded.var = 'concat_stage5' ;
lastAdded.depth = inputdepthbase + outputdepth1 + outputdepth2;

%% stage5
inputvarbase5 = lastAdded.var; %%important
inputdepthbase5 = lastAdded.depth;
lastAdded.var = inputvarbase5;
lastAdded.depth = inputdepthbase5;
add_conv_relu('M%s1_stage5_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s2_stage5_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s3_stage5_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s4_stage5_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s5_stage5_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s6_stage5_L1', 1, 1, 128, 'relu', true) ;
add_conv_relu('M%s7_stage5_L1', 1, 1, 38, 'relu', false) ;
outputvar1 = lastAdded.var;
outputdepth1 = lastAdded.depth;

lastAdded.var = inputvarbase5;
lastAdded.depth = inputdepthbase5;
add_conv_relu('M%s1_stage5_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s2_stage5_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s3_stage5_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s4_stage5_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s5_stage5_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s6_stage5_L2', 1, 1, 128, 'relu', true) ;
add_conv_relu('M%s7_stage5_L2', 1, 1, 19, 'relu', false) ;
outputvar2 = lastAdded.var;
outputdepth2 = lastAdded.depth;

net.addLayer('concat_stage6',...
    dagnn.Concat(), ...
    {outputvar1,outputvar2,inputvarbase}, 'concat_stage6');
lastAdded.var = 'concat_stage6' ;
lastAdded.depth = inputdepthbase + outputdepth1 + outputdepth2;

%% stage6
inputvarbase6 = lastAdded.var; %%important
inputdepthbase6 = lastAdded.depth;
lastAdded.var = inputvarbase6;
lastAdded.depth = inputdepthbase6;
add_conv_relu('M%s1_stage6_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s2_stage6_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s3_stage6_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s4_stage6_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s5_stage6_L1', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s6_stage6_L1', 1, 1, 128, 'relu', true) ;
add_conv_relu('M%s7_stage6_L1', 1, 1, 38, 'relu', false) ;
outputvar1 = lastAdded.var;
outputdepth1 = lastAdded.depth;

lastAdded.var = inputvarbase6;
lastAdded.depth = inputdepthbase6;
add_conv_relu('M%s1_stage6_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s2_stage6_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s3_stage6_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s4_stage6_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s5_stage6_L2', 7, 1, 128, 'relu', true) ;
add_conv_relu('M%s6_stage6_L2', 1, 1, 128, 'relu', true) ;
add_conv_relu('M%s7_stage6_L2', 1, 1, 19, 'relu', false) ;
outputvar2 = lastAdded.var;
outputdepth2 = lastAdded.depth;

net.addLayer('concat_stage7',...
    dagnn.Concat(), ...
    {outputvar2,outputvar1}, 'concat_stage7');
lastAdded.var = 'concat_stage7' ;
lastAdded.depth = outputdepth1 + outputdepth2;

net.initParams();
init_net();
netStruct = net.saveobj() ;
save('matconvnet_coco_openpose_model.mat', '-struct', 'netStruct') ;
clear netStruct ;


    function add_conv_relu(name_temp, ksize, stride, depth, varargin)
    % add a Convolutional + ReLU block
      args.relu = true ;
      args = vl_argparse(args, varargin) ;

      conv_name = sprintf(name_temp,'conv');
      pars = {[conv_name 'w'], [conv_name 'b']} ;
      net.addLayer(conv_name, ...
                   dagnn.Conv('size', [ksize ksize lastAdded.depth depth], ...
                              'stride', stride, ....
                              'pad', (ksize - 1) / 2, ...
                              'hasBias', true, ...
                              'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
                              lastAdded.var, conv_name, pars) ;
      lastAdded.var = conv_name;
      lastAdded.depth = depth ;
      if args.relu
          relu_name = sprintf(name_temp,'relu');
          net.addLayer(relu_name , dagnn.ReLU(), lastAdded.var, relu_name) ;
          lastAdded.var = relu_name ;
      end
    end

    function init_net()
        % load param
        pretained_param = load('openpose_param.mat');
        for i = 1:numel(net.params)
            param_temp = pretained_param.(net.params(i).name);
            fprintf('%s init size: %d with %d ',net.params(i).name, numel(param_temp), numel(net.params(i).value));
            assert(numel(param_temp) == numel(net.params(i).value), 'two inputs with same len');
            if net.params(i).name(end) == 'w'
                net.params(i).value = permute(param_temp,[2,1,3,4]);%filter
            else
                net.params(i).value = param_temp;%bias
            end
            fprintf('done!\n');
        end
        net.params(net.getParamIndex('conv1_1w')).value = net.params(net.getParamIndex('conv1_1w')).value(:,:,[3,2,1],:);
    end
    

end

