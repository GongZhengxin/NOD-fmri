% Code dependencies:
% - analyzePRF
% - knkutils (http://github.com/kendrickkay/knkutils/)
% - MatlabCIFTI
% - workbench
%
% make sure above dependencies are well located
addpath(genpath([pwd '/analyzePRF']));
addpath(genpath([pwd '/BIN_aperture']));
addpath(genpath([pwd '/cifti-matlab']));
addpath(genpath([pwd '/knkutils']));
addpath(genpath([pwd '/results']));

% define
datasetdir = '.'
path = [datasetdir '/derivatives/ciftify'];
sub = 'sub-02';
subjs = [path, sub, '/results'];
prfruns = [subjs '/*prf*'];
num_run = length(dir(prfruns));
all_runs = cell(num_run,1);
for iii = 1:num_run
    all_runs{iii} = ['ses-prf_task-prf_run-',...
        num2str(iii),'/ses-prf_task-prf_run-',...
        num2str(iii),'_Atlas_hp128_s4.dtseries.nii'];
end

% session fold
tr = 2;                % temporal sampling rate in seconds
pxtodeg = 16.0/200;    % conversion from pixels to degrees
wbcmd = 'wb_command';  % path to workbench command

% define which model fit to perform (1 through 3)
typ = 1;  % 1 is all runs, 2 is first half of each run, 3 is second half of each run

% load stimulus apertures 
all_aperturefiles = {'RETCCWTR2.mat' ...
    'RETMBTR2.mat' ...
    'RETCWTR2.mat' ...
    'RETMBTR2.mat' ...
    'RETEXPTR2.mat' ...
    'RETMBTR2.mat' ...
    'RETCONTR2.mat' ...
    'RETMBTR2.mat' ...
    'RETCONTR2.mat' ...
    'RETMBTR2.mat' ...
    'RETCCWTR2.mat' ...
    'RETMBTR2.mat' ...
    };

% define runs for analyze
selected_runs = {linspace(1, num_run, num_run)};

for iii=1:length(selected_runs)
    aperturefiles = all_aperturefiles(selected_runs{iii});
    runs = all_runs(selected_runs{iii});
    
    stimulus = {};
    for p=1:length(aperturefiles)
        a1 = load(aperturefiles{p},'stim');
        stimulus{p} = double(a1.stim);
        clear a1;
    end
    
    % load data
    data = {};
    for p=1:length(runs)
        data{p} = double(getfield(ciftiopen([subjs '/' runs{p}],wbcmd),'cdata'));
    end
    
    % deal with subsetting
    switch typ
        case 1
        case 2
            stimulus = cellfun(@(x) x(:,:,1:75),stimulus,'UniformOutput',0);
            data =     cellfun(@(x) x(:,1:75),  data,    'UniformOutput',0);
        case 3
            stimulus = cellfun(@(x) x(:,:,76:150),stimulus,'UniformOutput',0);
            data =     cellfun(@(x) x(:,76:150),  data,    'UniformOutput',0);
    end
    
    % load the voxel mask
    atlas = load('brain_mask_cortex.mat').roi_mask;%load('atlas.mat').glasser2016; load('MMP_mpmLR32k.mat').glasser_MMP;
    vxs = [find(atlas==1); find(atlas==4); find(atlas==5); find(atlas==6);find(atlas==7)];
    % fit the models
    try
        result = analyzePRF(stimulus,data,tr,struct('seedmode',2, 'vxs', vxs, 'maxpolydeg', 3, 'xvalmode', 0, 'wantsparse', 1));
        % prepare outputs
        quants = {'ang' 'ecc' 'gain' 'R2' 'rfsize'};
        allresults = zeros(91282,length(quants),'single');  % 91282 x 5
        allresults(:,1,typ) = result.ang(:,1);
        allresults(:,2,typ) = result.ecc(:,1)*pxtodeg;     % convert to degrees
        allresults(:,3,typ) = result.gain(:,1);
        allresults(:,4,typ) = result.R2(:,1);
        allresults(:,5,typ) = result.rfsize(:,1)*pxtodeg;  % convert to degrees
        % angle transformation
        angle = allresults(:,1);
        angle(angle<=180) = angle(angle<=180)/180*pi;
        angle(angle>180) = (angle(angle>180)-360)/180*pi;
        allresults(:,1) = angle;
        time = clock;
    catch
        time = clock;
    end
end
mkdir([subjs, '/ses-prf_task-prf/'])
save([subjs, '/ses-prf_task-prf/',  sub, '_retinotopy-allrun-s4_params.mat'], 'result'); 
