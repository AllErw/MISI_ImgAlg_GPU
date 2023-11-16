function img = DnS_1rec_fixed_pos_GPU_dll(rf_data,source_locations,receiver_location,image_coordinates,c,fsamp,CUDAparams)
% USAGE: 
% 
%   img = DnS_1rec_fixed_pos_GPU_dll(rf_data,source_locations,receiver_location,image_coordinates,c,fsamp,CUDAparams)
% 
% INPUTS:
%   - rf_data:          [Nt x Nsrc] matrix containing the RF data. 
%                       Nt is number of time samples, Nsrc number of source locations.
%   - source_locations: [Nsrc x 3] matrix containing, on each row, the x-,
%                       y- and z-coordinates [m] of the source locations.
%   - receiver_location:[1 x 3] vector containing the coordinates of the receiver.
%   - image_coordinates:[Nimg x 3] matrix containing the coordinates of
%                       all Nimg image grid points.
%   - c:                speed of sound [m/s].
%   - fsamp:            RF sample frequency [Hz].
%   - CUDAparams:       (OPTIONAL) [2 x 1] vector: [# of threads per block , # of blocks]
%                                  default = [1024 , (Nimg+1024-1) / 1024].
% 
% OUTPUT:
%   - img:              [Nimg x 1] matrix containing the raw image data,
%                       i.e., linear scale, non-envelope-detected.

if ~libisloaded('MISI_ImgAlg_GPU')
    disp('Loading library...');
    evalc('[notfound,warnings] = loadlibrary(''MISI_ImgAlg_GPU.dll'',''MISI_ImgAlg_GPU.h'');');
    disp('Library loaded.');
end

if (size(source_locations,2)~=3 || numel(receiver_location)~=3 || size(image_coordinates,2)~=3 || size(rf_data,2)~=size(source_locations,1))
    disp('Error in dimensions of input matrices. Dimensions should be: ');
    disp('    rf_data:              Nt x Nsrc');
    disp('    source_locations:     Nsrc x 3');
    disp('    receiver_location:    1 x 3');
    disp('    image_coordinates:    Nimg x 3');
end

c = single(abs(real(c)));
fsamp = single(abs(real(fsamp)));

[Nt,Nsrc] = size(rf_data);
Nimg = size(image_coordinates,1);
image = zeros(Nimg,1,'single');

if nargin==6
%     CUDAparams = int32([1024,(Nimg+1024-1) / 1024]);
    CUDAparams = int32([1024,128]);
end

[~,~,~,~,~,img] = calllib('MISI_ImgAlg_GPU','DnS_1rec_fixed_pos_GPU_chunks_interface',...
                          rf_data,source_locations,receiver_location,image_coordinates,c,fsamp,Nsrc,Nt,Nimg,CUDAparams,image);
