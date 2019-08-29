load('testdata.mat')
data={};
data{1} = X;
spiketrain = double(spiketrain);
data{2} = spiketrain;
modelType=[1 1 1 0];
param = param';
[f,df,hess]=ln_poisson_model(param,data,modelType);

%%
data{1}= X(1:10000,:);
data{2} = spiketrain(1:10000,:);
opts = optimset('Gradobj','on','Hessian','on','Display','off');
[param,fval] = fminunc(@(param) ln_poisson_model(param,data,modelType),param,opts);

%%
hd_vector = 2*pi/n_dir_bins/2:2*pi/n_dir_bins:2*pi - 2*pi/n_dir_bins/2;
theta_vector = hd_vector;
speed_vector = 2.5:50/n_speed_bins:47.5;

n_pos_bins = 20;
n_dir_bins = 18;
n_speed_bins = 10;
% show parameters from the full model
param_full_model = param;
theta_param = [0];

% pull out the parameter values
pos_param = param_full_model(1:n_pos_bins^2);
hd_param = param_full_model(n_pos_bins^2+1:n_pos_bins^2+n_dir_bins);
speed_param = param_full_model(n_pos_bins^2+n_dir_bins+1:n_pos_bins^2+n_dir_bins+n_speed_bins);
% theta_param = param_full_model(numel(param_full_model)-n_theta_bins+1:numel(param_full_model));

% compute the scale factors
% NOTE: technically, to compute the precise scale factor, the expectation
% of each parameter should be calculated, not the mean.
scale_factor_pos = mean(exp(speed_param))*mean(exp(hd_param))*mean(exp(theta_param))*50;
scale_factor_hd = mean(exp(speed_param))*mean(exp(pos_param))*mean(exp(theta_param))*50;
scale_factor_spd = mean(exp(pos_param))*mean(exp(hd_param))*mean(exp(theta_param))*50;
% scale_factor_theta = mean(exp(speed_param))*mean(exp(hd_param))*mean(exp(pos_param))*50;

% compute the model-derived response profiles
pos_response = scale_factor_pos*exp(pos_param);
hd_response = scale_factor_hd*exp(hd_param);
speed_response = scale_factor_spd*exp(speed_param);
% theta_response = scale_factor_theta*exp(theta_param);

% plot the model-derived response profiles
subplot(3,4,5)
imagesc(reshape(pos_response,20,20)); axis off; 
subplot(3,4,6)
plot(hd_vector,hd_response,'k','linewidth',3)
xlabel('direction angle')
box off
subplot(3,4,7)
plot(speed_vector,speed_response,'k','linewidth',3)
xlabel('Running speed')
box off
% subplot(3,4,8)
% plot(theta_vector,theta_response,'k','linewidth',3)
% xlabel('Theta phase')
% axis([0 2*pi -inf inf])
% box off

