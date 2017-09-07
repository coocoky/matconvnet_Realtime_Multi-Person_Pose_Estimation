addpath /home/qwang/caffe/matlab % path of your <caffe_matlab>
model = './pose_deploy_linevec.prototxt';
weights = './pose_iter_440000.caffemodel';

caffe.set_mode_cpu();
net = caffe.Net(model, weights, 'test'); % create net and load weights

layer_names = net.layer_names;

com3 = ['save(''openpose_param.mat'',''layer_names'','];
for layer_name = layer_names'
	layer = layer_name{1};
	if ~isempty(strfind(layer, 'conv')) && isempty(strfind(layer,'relu'))
		com1 = [layer 'w=net.params(''' layer ''',1).get_data();'];
		com2 = [layer 'b=net.params(''' layer ''',1).get_data();'];
		com3 = [com3 '''' layer 'w'',''' layer 'b'','];
		disp(com1);eval(com1)
		disp(com2);eval(com2)
	end
end

com3(end) = [];
com3 = [com3 ');'];
disp(com3);eval(com3)
