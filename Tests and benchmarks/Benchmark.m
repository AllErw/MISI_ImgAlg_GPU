% Benchmark CPU vs GPU

% Check if library is loaded; do so if not.
freshly_loaded = 0;
if ~libisloaded('MISI_CPU')
    warning off;
    loadlibrary('C:\Users\Erwin Alles\Documents\GitHub\MISI_ImgAlg\x64\Release\MISI_ImgAlg.dll',...
                'C:\Users\Erwin Alles\Documents\GitHub\MISI_ImgAlg\MISI_ImgAlg.h',...
                'alias','MISI_CPU');
    warning on;
    disp('Library loaded.');
    libfunctionsview('MISI_CPU');
    freshly_loaded = freshly_loaded+1;
end
if ~libisloaded('MISI_GPU')
    warning off;
    loadlibrary('C:\Users\Erwin Alles\Documents\GitHub\MISI_ImgAlg_GPU\MISI_ImgAlg_GPU.dll',...
                'C:\Users\Erwin Alles\Documents\GitHub\MISI_ImgAlg_GPU\MISI_ImgAlg_GPU.h',...
                'alias','MISI_GPU');
    warning on;
    disp('Library loaded.');
    libfunctionsview('MISI_GPU');
    freshly_loaded = freshly_loaded+1;
end
if freshly_loaded>0;  return;     end

%% Set parameters and generate RF data:
METHOD = 2;     % Flag for reconstruction: 1 = DAS, 2 = DMAS

% Load RF data:
load('test_data.mat');
Nsrc = data.Npos;  Nt = length(data.taxis);
c = data.soundspeed; fsamp = data.fsamp;
rf_data = data.RFdata';
receiver_location = data.hydrophone;
source_locations = data.sourcecoors;

delta = [10 20 50 100]*1E-6;%[2 5 10 20 50 100]*1E-6;
delta = delta(end:-1:1);

Npix = zeros(size(delta));
timeGPU = zeros(size(delta));
timeCPU = zeros(size(delta));

figure;
for dcnt = 1:length(delta)
    % Set parameters:
    xaxis           = -8E-3 : delta(dcnt) : 8E-3;
    yaxis           =  0;
    zaxis           =  0E-3 : delta(dcnt) : 12E-3;
    Nx = length(xaxis);   Ny = length(yaxis);   Nz = length(zaxis);
    [X,Y,Z] = meshgrid(xaxis  ,  yaxis  ,  zaxis);
    X = reshape(X,numel(X),1);Y = reshape(Y,numel(Y),1);Z = reshape(Z,numel(Z),1);
    image_coordinates = [X Y Z];
    Nimg = length(X);
    image = zeros(Nimg,1,'single');
    
    Npix(dcnt) = Nimg;


    %% Perform the actual benchmarking:

    % *** CPU: ***
    tic;
    for aa = 1:(10 - 9*(Nimg>1E5)) % to disable averaging when Nimg >100k
        switch METHOD
            case 1
                [~,~,~,~,imgCPU] = calllib('MISI_CPU','DnS_1rec_fixed_pos',...
                                  rf_data,source_locations,receiver_location,image_coordinates,c,fsamp,Nsrc,Nt,Nimg,image);
            case 2
                [~,~,~,~,imgCPU] = calllib('MISI_CPU','DMnS_1rec_fixed_pos',...
                                  rf_data,source_locations,receiver_location,image_coordinates,c,fsamp,Nsrc,Nt,Nimg,image);
        end
    end
    tCPU = toc/aa;
    timeCPU(dcnt) = tCPU;

    % *** GPU: ***
    switch METHOD
        case 1
            CUDAparams = int32([1024,(Nimg+1024-1) / 1024]);
%             CUDAparams = int32([1024,1]);
        case 2
            CUDAparams = int32([512,1]);
%             CUDAparams = int32([1024,1]);
    end
    tic;
    for aa = 1:(10 - 9*(Nimg>1E5)) % to disable averaging when Nimg >100k
        switch METHOD
            case 1
                [~,~,~,~,~,imgGPU] = calllib('MISI_GPU','DnS_1rec_fixed_pos_GPU_chunks_interface',...
                                     rf_data,source_locations,receiver_location,image_coordinates,c,fsamp,Nsrc,Nt,Nimg,CUDAparams,image);
            case 2
                [~,~,~,~,~,imgGPU] = calllib('MISI_GPU','DMnS_1rec_fixed_pos_GPU_chunks_interface',...
                                     rf_data,source_locations,receiver_location,image_coordinates,c,fsamp,Nsrc,Nt,Nimg,CUDAparams,image);
        end
    end
    tGPU = toc/aa;
    timeGPU(dcnt) = tGPU;

%     fprintf('%3.1E pixels, CPU time: %5.3f s, GPU: %5.3f s.\n',Nimg,tCPU,tGPU);
%     fprintf('Difference between CPU and GPU: %5.3f%%\n',100*sum(abs(img-imgGPU)) / sum(abs(img)));
    loglog(Npix , [timeCPU ; timeGPU]);
    legend('CPU','GPU','location','northwest');
    xlabel('Number of pixels');
    ylabel('Wall clock time [s]');
    drawnow;
end
title(['Speed-up: ca. ',num2str(tCPU/tGPU,3),' times']);
axis([1E4 1E8 1E-3 1E3]);
