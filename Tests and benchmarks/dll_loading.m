if ~libisloaded('IntDevImgAlgGPU')
    warning off;
        loadlibrary('IntDevImgAlgGPU.dll','IntDevImgAlgGPU.h');
    warning on;
    disp('Libray loaded.');
%     libfunctionsview('IntDevImgAlgGPU');
    return;
end

METHOD = 2;     % Flag for reconstruction: 1 = DAS, 2 = DMAS

% Set parameters and load test data:
xaxis           = -8E-3 : 20E-6 : 8E-3;
yaxis           =  0;
zaxis           =  0E-3 : 20E-6 : 12E-3;
Nx = length(xaxis);   Ny = length(yaxis);   Nz = length(zaxis);
[X,Y,Z] = meshgrid(xaxis  ,  yaxis  ,  zaxis);
X = reshape(X,numel(X),1);Y = reshape(Y,numel(Y),1);Z = reshape(Z,numel(Z),1);

load('test_data.mat');
Nsrc = data.Npos;  Nt = length(data.taxis);
c = data.soundspeed; fsamp = data.fsamp;
rf_data = data.RFdata';
receiver_location = data.hydrophone;
source_locations = data.sourcecoors;
image_coordinates = [X Y Z];
Nimg = length(X);
image = zeros(Nimg,1,'single');

% *** CPU TEST: ***
tic;
for aa = 1:(10 - 9*(Nimg>1E5)) % to disable averaging when Nimg >100k
    switch METHOD
        case 1
            img = DnS_1rec_fixed_position_dll_no_envelope(rf_data,source_locations,receiver_location,image_coordinates,c,fsamp,1.0);
        case 2
            img = DMnS_1rec_fixed_position_mex(rf_data,source_locations,receiver_location,image_coordinates,c,fsamp,1);
    end
end
tCPU = toc/aa;

% *** GPU TEST: ***
switch METHOD
    case 1
        CUDAparams = int32([1024,(Nimg+1024-1) / 1024]);
    case 2
        CUDAparams = int32([1024,1]);
end
tic;
for aa = 1:(10 - 9*(Nimg>1E5)) % to disable averaging when Nimg >100k
    switch METHOD
        case 1
%             [~,~,~,~,~,imgGPU] = calllib('IntDevImgAlgGPU','DnS_1rec_fixed_pos_GPU_interface',...
%                               rf_data,source_locations,receiver_location,image_coordinates,c,fsamp,Nsrc,Nt,Nimg,CUDAparams,image);
            [~,~,~,~,~,imgGPU] = calllib('IntDevImgAlgGPU','DnS_1rec_fixed_pos_GPU_chunks_interface',...
                              rf_data,source_locations,receiver_location,image_coordinates,c,fsamp,Nsrc,Nt,Nimg,CUDAparams,image);
        case 2
%             [~,~,~,~,~,imgGPU] = calllib('IntDevImgAlgGPU','DMnS_1rec_fixed_pos_GPU_interface',...
%                                  rf_data,source_locations,receiver_location,image_coordinates,c,fsamp,Nsrc,Nt,Nimg,CUDAparams,image);
            [~,~,~,~,~,imgGPU] = calllib('IntDevImgAlgGPU','DMnS_1rec_fixed_pos_GPU_chunks_interface',...
                                 rf_data,source_locations,receiver_location,image_coordinates,c,fsamp,Nsrc,Nt,Nimg,CUDAparams,image);
    end
end
tGPU = toc/aa;

% *** Attempt at concurrent calls using partial data: SLOWER!!! ***
% tic;
% NperThread = ceil(Nimg/10);
% images = zeros(NperThread,10);
% parfor aa = 1:9
%     range = (1:NperThread) + (aa-1)*NperThread;
%     IC = image_coordinates(range,:);
%     images(:,aa) = DnS_1rec_fixed_pos_GPU_dll(rf_data,source_locations,receiver_location,IC,c,fsamp)
% end
% images = reshape(images,numel(images),1);
% tGPU = toc;


% ***PRECOMPUTATION OF DELAYS ETC. IS MUCH SLOWER!!! ***
% delays = zeros(Nimg*Nsrc,1,'uint16');
% % [~,~,~,delays] = calllib('ImgAlg','DnS_1rec_fixed_pos_precomp',...
% %                          source_locations,receiver_location,image_coordinates,c,fsamp,Nsrc,Nimg,delays);
% % [~,~,imgGPU] = calllib('ImgAlg','DnS_1rec_fixed_pos_from_precomp',...
% %                        rf_data,delays,Nsrc,Nt,Nimg,image);
% [~,~,~,delays] = calllib('IntDevImgAlgGPU','DnS_1rec_fixed_pos_precomp_delays_GPU_interface',...
%                          source_locations,receiver_location,image_coordinates,c,fsamp,Nsrc,Nt,Nimg,delays);
% [~,~,imgGPU] = calllib('IntDevImgAlgGPU','DnS_1rec_fixed_pos_from_precomp_GPU_interface',...
%                      rf_data,delays,Nsrc,Nt,Nimg,image);


fprintf('%3.1E pixels, CPU time: %5.3f s, GPU: %5.3f s.\n',Nimg,tCPU,tGPU);
fprintf('Difference between CPU and GPU: %5.3f%%\n',100*sum(abs(img-imgGPU)) / sum(abs(img)));

if exist('data','var')
    img    = squeeze(reshape(img,   Nx,Ny,Nz));
    imgGPU = squeeze(reshape(imgGPU,Nx,Ny,Nz));
    switch METHOD
        case 1
            img = abs(hilbert(img'));
            imgGPU = abs(hilbert(imgGPU'));
        case 2
            img = abs(img)';        img    = conv2(img,   ones(3,1),'same');
            imgGPU = abs(imgGPU)';  imgGPU = conv2(imgGPU,ones(3,1),'same');
    end
    
    figure;
    subplot(2,2,1);
    imagesc(xaxis*1000,zaxis*1000,img)
    axis equal tight;

    subplot(2,2,2);
    imagesc(xaxis*1000,zaxis*1000,imgGPU)
    axis equal tight;

    subplot(2,2,[3 4]);
    imagesc(xaxis*1000,zaxis*1000,img-imgGPU)
    axis equal tight;

end

return;

%%
if libisloaded('IntDevImgAlgGPU')
    warning off;
        unloadlibrary('IntDevImgAlgGPU');
    warning on;
    disp('Library unloaded.');
end