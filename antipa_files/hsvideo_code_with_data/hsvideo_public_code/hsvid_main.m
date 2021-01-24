% Change MATLAB path to this script's location and execute. It will create
% folders using relative paths to save data. Outputs will be in ../recons/
% relative to data_path.
% PSF file should be in ../hsvideo_public_data/PSF/[psf_file_name]
% You can change raw data path, but this assumes ../hsvideo_public_data/raw

% Read in raw data
data_path = '../hsvideo_public_data/raw';
file_name = 'tennis_bounce_00182.tif';   %Raw data file name. This file needs to exist in data_path

psf_path = '../hsvideo_public_data/PSF/psf_averaged_2018-12-5.tif';   %PSF path

params.psf_bias =103;
params.ds_t = 2;   %Temporal downsampling. Use 1 or 2. ds_t = 2 will reduce time points by 2x. Useful for fitting in GPU
params.meas_bias = 103;
init_style = 'zeros';   %Use 'loaded' to load initialization, 'zeros' to start from scratch. Admm will run 2D deconv, then replicate result to all time points
params.ds = 8;   %Lateral downsampling. Use 4 or 8
params.nlines = 11;   %Determined by combination of line time ane exposure time.
params.colors = [1,2,3];     %1 is red, 2 is green, 3 is blue.
useGpu = 0;      %Boolean--0 to use CPU, 1 to use GPU.


%psf = imresize(warp_im(double(imread(psf_path))- params.psf_bias,1.00001), 1/params.ds, 'box'); 

% Remove bias from PSF, then resize (the bias term depends on your camera!
% Our prorotype has a bias of approximately 100 counts)
psf = imresize(double(imread(psf_path))- params.psf_bias, 1/params.ds, 'box'); 
h = psf/max(sum(sum(psf)));  %Normalize the PSF

% Read in data
data_in = imresize(double(imread([data_path,'/', file_name])) - params.meas_bias, 1/params.ds, 'box');
b = data_in/max(data_in(:));

Nx = 2*size(h,2);  % Compute the problem size based on PSF
Ny = 2*size(h,1);

%define crop and pad operators to handle 2D fft convolution
pad2d = @(x)padarray(x,[size(h,1)/2,size(h,2)/2],0,'both');

cc = uint16((size(h,2)/2+1):3*size(h,2)/2);
rc = uint16((size(h,1)/2+1):3*size(h,1)/2);
if useGpu
    cc = gpuArray(cc);
    rc = gpuArray(rc);
end
crop2d = @(x)x(rc,cc);

shutter_indicator = hsvid_crop([Ny/2, Nx/2],params.nlines, pad2d);   % Creates the rolling shutter indicator function in the padded 3d space

% Allows separate temporal downsampling--this is useful if you want higher
% lateral resolution but don't want to oversample in time. This is
% important for fitting in a GPU.

% shutter_indicator holds the pointwise multiplication function (S(y,t) in
% the paper)

Nt = size(shutter_indicator,3);
if params.ds_t == 2
    if floor(Nt/2)*2 ~= Nt
        shutter_end = shutter_indicator(:,:,end);
        shutter_indicator = shutter_indicator(:,:,1:end-1);
    end
    shutter_indicator = .5*shutter_indicator(:,:,1:2:floor(Nt/2)*2) + .5*shutter_indicator(:,:,2:2:floor(Nt/2)*2);
    if mod(size(shutter_indicator,3)/2,1)~=0
        %shutter_indicator = cat(3,shutter_indicator,shutter_end);
        shutter_indicator(:,:,end-1) = shutter_indicator(:,:,end-1) + shutter_indicator(:,:,end);
        shutter_indicator = shutter_indicator(:,:,1:end-1);   %Make it even
    end
end

% Get the new size of the shutter indicator.
Nt = size(shutter_indicator,3);
if useGpu
    shutter_indicator = gpuArray(single(shutter_indicator));
end

% Initialize the video. the option 'admm' requires the admm2d_solver from
% antipa/lensless. You can just use 'zeros' -- it's pretty good. 

if strcmpi(init_style, 'zeros')
    xinit_rgb = zeros(Ny, Nx, Nt,3);
elseif strcmpi(init_style,'loaded')
    xinit = imnormalized(:,:,:,cindex);
elseif strcmpi(init_style,'admm')
    xinit_2d = gpuArray(single(zeros(Ny, Nx, 3))); 
    xinit_rgb = zeros(Ny, Nx, Nt,3);
    for n = 1:3
        xinit_2d(:,:,n) = admm2d_solver(gpuArray(single(b(:,:,n))), gpuArray(single(h(:,:,n))),[],.001); 
        xinit_rgb(:,:,:,n) = params.nlines*repmat(gather(xinit_2d(:,:,n)),[1,1,Nt]);
        imagesc(2*xinit_2d/max(xinit_2d(:)))
    end
end

%axis image
clear xinit_2d  %if xinit_2d was created during admm initialization, delete it to save memory.
%%

options.color_map = 'parula';  %You can change the colormap--each color is solved independently so they appear grayscale during solving.

% These are manually entered stepsizes. In practice, this should be
% replaced with something smart the estimates the step size using power
% iterations.

if params.ds == 8
    if params.nlines == 22
        options.stepsize = .3;
    elseif params.nlines == 6
        %options.stepsize = 4;
        options.stepsize = 4;

    elseif params.nlines == 3
        %options.stepsize = 8;
        options.stepsize = 24;
    elseif params.nlines == 1
        options.stepsize = 100;
    elseif params.nlines == 11
        if params.ds_t == 2
            options.stepsize = 3;   %<----- this is the default value!
        else
            options.stepsize = 1.5;
        end
    else
        options.stepsize = 1.5;
    end
elseif params.ds == 12
    options.stepsize = 1;  %1 works for hande
elseif params.ds == 4
    if params.nlines == 1
        options.stepsize = 12;
    elseif params.nlines == 11
        options.stepsize = 3.5;
    elseif params.nlines == 22
        options.stepsize = 7;
    end
end

% These are options used by proxMin -- if you are using your own solver,
% you don't need these.
options.convTol = 15e-10;

%options.xsize = [256,256];
options.maxIter = 2000;    %Max iterations of FISTA
options.residTol = 5e-5;   %If residual is below this, stop
options.momentum = 'nesterov';   %Leave as nesterov! 
options.disp_figs = 1;    %Boolean -- 1 to show figs during solving, 0 to not plot anything.
options.disp_fig_interval = 5;   %display image this often
options.xsize = size(h);   %Store dimensions of solution
options.print_interval = 1;    %Displays status of solving at commandline if you like numbers

% Create figure handle 
h1 = figure(1);
clf
options.fighandle = h1;
nocrop = @(x)x;
options.known_input = 0;   %If you have groundtruth, you can set this to 1 and it'll compare to ground truth as it goes -- not tested so no guarantees!
xhat_rgb = zeros(Ny, Nx, Nt, 3);    % Initialize the solution
%%
% Loop over colors
for cindex = params.colors
    % Using single precision is good enough and goes 2x as fast
    if useGpu
        H = gpuArray(single(fft2(ifftshift(pad2d(h(:,:,cindex))))));
    else
        H = single(fft2(ifftshift(pad2d(h(:,:,cindex)))));
    end
    Hconj = conj(H);


    % Setup forward op
    A = @(x)hsvidA(x, H, shutter_indicator, crop2d);
    % Adjoint
    Aadj = @(y)hsvidAadj(y, Hconj, shutter_indicator, pad2d);
    % Gradient

    if useGpu
        grad_handle = @(x)linear_gradient_b(x, A, Aadj, gpuArray(single(b(:,:,cindex))));
    else
        grad_handle = @(x)linear_gradient_b(x, A, Aadj, b(:,:,cindex));
    end
    
    %Prox
   % prox_handle = @(x)deal(x.*(x>=0), abs(sum(sum(sum(x(x<0))))));
   
    params.tau1 = single(.000006);   %Anisotropic TV parameter. .000005 works pretty well for v1 camera, .0002 for v2
 
    params.t_weighting = single(30);    %Weighting for temporal dimension. This should be typically much higher than lateral weighting.
    % Parameters if using iterative TV for isotropic regularization
    
    params.tau_iso = single(.25e-4);     
    TVpars.epsilon = 1e-7;
    TVpars.MAXITER = 100;
    TVpars.alpha = .3;
    

    if useGpu
        TVpars.epsilon = gpuArray(single(TVpars.epsilon));
        TVpars.MAXITER = gpuArray(single(TVpars.MAXITER));
        TVpars.alpha = gpuArray(single(TVpars.alpha));
        params.tau1 = gpuArray(single(params.tau1));
        params.tau_iso = gpuArray(single(params.tau_iso));
        params.t_weighting = gpuArray(single(params.t_weighting));
    end
    
    prox_handle = @(x)deal(1/2*(max(x,0) + tv3dApproxHaar(x, params.tau1, params.t_weighting)), params.tau1*hsvid_TVnorm3d(x));   % This does does averaging of anisotropic TV with nonnegativity to get nonnegative TV results
    %prox_handle = @(x)deal(hsvid_TV3DFista(x, tau_iso, 0, 9999999, TVpars) , tau_iso*hsvid_TVnorm3d(x));
    % ^^^Uncomment this if you want to use isotropic iterative TV with values
    % constrained to nonnegative. This is the most accurate, but is very
    % slow. 
   
    switch lower(init_style)
        case('zeros')
            xinit = zeros(Ny, Nx, Nt);
        case('admm')
            xinit = squeeze(xinit_rgb(:,:,:,cindex));
    end
      
    
    if useGpu
        xinit = gpuArray(single(xinit));
        [xhat, f2] = proxMin(grad_handle,prox_handle,xinit,gpuArray(single(b(:,:,cindex))),options);
    else   %This uses single precision for speed
        [xhat, f2] = proxMin(grad_handle,prox_handle,single(xinit),single(b(:,:,cindex)),options);
    end
    xhat_rgb(:,:,:,cindex) = gather(xhat);
    clear xhat
end

%%


datestamp = datetime;
date_string = datestr(datestamp,'yyyy-mmm-dd_HHMMSS');
save_str = ['../recons/',date_string,'_',file_name(1:end-4)];
full_path = fullfile(data_path,save_str);
mkdir(full_path);


%%
imout = xhat_rgb/prctile(xhat_rgb(:),99.99);
imbase = file_name(1:end-4);
mkdir([full_path, '/png/']);
filebase = [full_path, '/png/', imbase];
out_names = {};
for n= 1:size(imout,3)
    out_names{n} = [filebase,'_',sprintf('%.3i',n),'.png'];
    imwrite(squeeze(imout(:,:,n,:)),out_names{n});
    fprintf('writing image %i of %i\n',n,size(xhat_rgb,3))
end

fprintf('zipping...\n')
zip([full_path, '/png/', imbase],out_names)
fprintf('done zipping\n')
%%
fprintf('writing .mat\n')
save([full_path,'/',file_name(1:end-4),'_',date_string,'.mat'], 'tau_iso','TVpars','xhat_rgb', 'options', 'h', 'b','params','options','-v7.3')
fprintf('done writing .mat\n')
