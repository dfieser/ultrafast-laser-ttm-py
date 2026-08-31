function generate_fixtures()
%GENERATE_FIXTURES  Golden-fixture generation for the laserttm Python port.
%
%   Runs the MATLAB reference solvers from ultrafast-laser-ttm-toolbox
%   (expected as a sibling directory) on a fixed set of configurations and
%   saves each result struct twice:
%     fixtures/<case>.mat   full-precision, authoritative (-v7, scipy-readable)
%     fixtures/<case>.json  human-readable companion (NaN/Inf become null)
%   plus fixtures/manifest.json with MATLAB version and per-case wall times.
%
%   The Python test suite asserts agreement with these fixtures within
%   stated tolerances. Wall times double as the performance baseline.
%
%   Run from this folder:   matlab -batch generate_fixtures

thisDir     = fileparts(mfilename('fullpath'));
matlabRepo  = fullfile(thisDir, '..', '..', 'ultrafast-laser-ttm-toolbox');
fixturesDir = fullfile(thisDir, 'fixtures');
assert(exist(fullfile(matlabRepo, 'src'), 'dir') == 7, ...
    'MATLAB reference repo not found at %s', matlabRepo);
addpath(fullfile(matlabRepo, 'src'));
if ~exist(fixturesDir, 'dir'); mkdir(fixturesDir); end

manifest = struct();
manifest.matlabVersion   = version;
manifest.toolboxVersion  = strtrim(fileread(fullfile(matlabRepo, 'VERSION')));
manifest.generatedBy     = 'validation/generate_fixtures.m';
manifest.sourceRepo      = 'https://github.com/dfieser/ultrafast-laser-ttm-toolbox';
manifest.note            = ['.mat files are the authoritative full-precision ' ...
    'fixtures; in the JSON companions NaN/Inf serialize as null.'];
manifest.cases           = {};

caseList = buildCaseList(fixturesDir);

for i = 1:numel(caseList)
    c = caseList{i};
    fprintf('\n################ FIXTURE CASE %d/%d: %s ################\n', ...
        i, numel(caseList), c.name);
    entry = struct('name', c.name, 'solver', c.solver, 'status', 'ok', ...
        'wallTime_s', NaN, 'errorMessage', '');
    try
        t0 = tic;
        switch c.solver
            case 'Scanning_Beam_Solver'
                results = Scanning_Beam_Solver(c.cfg, c.outputDir, false);
            otherwise
                fh = str2func(c.solver);
                results = fh(c.cfg);
        end
        entry.wallTime_s = toc(t0);

        fx = struct();
        fx.caseName      = c.name;
        fx.solver        = c.solver;
        fx.matlabVersion = version;
        fx.wallTime_s    = entry.wallTime_s;
        fx.cfg           = sanitize(c.cfg);
        fx.results       = sanitize(results);
        save(fullfile(fixturesDir, [c.name '.mat']), 'fx', '-v7');
        writeJson(fullfile(fixturesDir, [c.name '.json']), fx);
        fprintf('>>> CASE OK: %s  (%.2f s)\n', c.name, entry.wallTime_s);
    catch ME
        entry.status = 'error';
        entry.errorMessage = ME.message;
        fprintf(2, '>>> CASE FAILED: %s\n%s\n', c.name, ...
            getReport(ME, 'extended', 'hyperlinks', 'off'));
    end
    manifest.cases{end+1} = entry;
    writeJson(fullfile(fixturesDir, 'manifest.json'), manifest);  % persist progress
end

fprintf('\n=== All %d cases attempted; manifest written. ===\n', numel(caseList));
end


% =========================================================================
function cases = buildCaseList(fixturesDir)
cases = {};
outDir = @(name) fullfile(fixturesDir, 'solver_outputs', name);

% Shared laser settings of the depth/radial/scanning example family
W40 = struct('material', 'W', 'Pavg', 40, 'spotRadius', 100e-6, ...
    'f_rep', 18e6, 'tau_FWHM', 500e-15, 'absorbance', 0.55, ...
    'makePlots', false, 'saveFigures', false);

% ---- Surface point (examples/Example_Surface_Point_Baseline.m) ----------
cfg = struct('material', 'W', 'Pavg', 10, 'spotRadius', 100e-6, ...
    'f_rep', 5e6, 'tau_FWHM', 500e-15, 'absorbance', 0.55, ...
    'makePlots', false, 'saveFigures', false);
cfg.simDuration = 50 / cfg.f_rep;
cfg.outputDir   = outDir('surface_point_baseline');
cases{end+1} = mkcase('surface_point_baseline', 'Surface_Point_Solver', cfg);

c2 = cfg; c2.material = 'Cu'; c2.outputDir = outDir('surface_point_cu');
cases{end+1} = mkcase('surface_point_cu', 'Surface_Point_Solver', c2);

c3 = cfg; c3.pulseProfile = 'square'; c3.outputDir = outDir('surface_point_square');
cases{end+1} = mkcase('surface_point_square', 'Surface_Point_Solver', c3);

% ---- Single pulse (solver defaults at the W example laser settings) -----
cfg = W40; cfg.outputDir = outDir('single_pulse_baseline');
cases{end+1} = mkcase('single_pulse_baseline', 'Single_Pulse_Visualizer', cfg);

% ---- Inversion quantifier (20 pulses, W example laser settings) ---------
cfg = W40; cfg.simDuration = 20 / cfg.f_rep;
cfg.outputDir = outDir('inversion_baseline');
cases{end+1} = mkcase('inversion_baseline', 'Inversion_Quantifier', cfg);

% ---- Depth profile: small fast case (radial derivation ON) --------------
cfg = W40; cfg.simDuration = 10 / cfg.f_rep;
cfg.enableRadialProfile = true;
cfg.outputDir = outDir('depth_profile_small');
cases{end+1} = mkcase('depth_profile_small', 'Depth_Profile_Solver', cfg);

% ---- Depth profile: examples/Example_Depth_Profile_Baseline.m -----------
cfg = W40; cfg.simDuration = 100 / cfg.f_rep;
cfg.enableRadialProfile = false;
cfg.outputDir = outDir('depth_profile_baseline');
cases{end+1} = mkcase('depth_profile_baseline', 'Depth_Profile_Solver', cfg);

% ---- Radial profile: small fast case ------------------------------------
cfg = W40; cfg.simDuration = 20 / cfg.f_rep;
cfg.Nr = 40; cfg.rMax_factor = 4; cfg.radialSolveMode = 'scale';
cfg.outputDir = outDir('radial_profile_small');
cases{end+1} = mkcase('radial_profile_small', 'Radial_Profile_Solver', cfg);

% ---- Radial profile: examples/Example_Radial_Profile_Baseline.m ---------
cfg = W40; cfg.simDuration = 100 / cfg.f_rep;
cfg.Nr = 80; cfg.rMax_factor = 4; cfg.radialSolveMode = 'scale';
cfg.outputDir = outDir('radial_profile_baseline');
cases{end+1} = mkcase('radial_profile_baseline', 'Radial_Profile_Solver', cfg);

% ---- Scanning beam: reduced case first ----------------------------------
p = scanningBaselineParams();
p.scanLength = 0.2e-3; p.Nx = 60; p.Ny = 30;
cases{end+1} = mkscan('scanning_small', p, outDir('scanning_small'));

% ---- Scanning beam: examples/Example_Scanning_Beam_Baseline.m -----------
p = scanningBaselineParams();
cases{end+1} = mkscan('scanning_baseline', p, outDir('scanning_baseline'));
end


function p = scanningBaselineParams()
% Verbatim from examples/Example_Scanning_Beam_Baseline.m
p = struct();
p.material     = 'W';
p.gamma        = 137.3;
p.Cl           = 2.54e6;
p.G            = 1.65e17;
p.kl           = 174;
p.Pavg         = 40;
p.spotRadius   = 100e-6;
p.f_rep        = 18e6;
p.tau_FWHM     = 100e-15;
p.pulseProfile = 'gaussian';
p.v_scan       = 1.0;
p.scanLength   = 2e-3;
p.absorbance   = 0.55;
p.Leff         = 100e-9;
p.T0_C         = 25;
p.Nx           = 120;
p.Ny           = 60;
p.xPad         = 3;
p.yExtent      = 5;
p.depthProfile = 'exponential';
p.dzTarget     = 500e-9;
p.Ndiff        = 100;
p.NadiPerGap   = 10;
end


function c = mkcase(name, solver, cfg)
c = struct('name', name, 'solver', solver, 'cfg', cfg, 'outputDir', '');
end

function c = mkscan(name, params, outputDir)
c = struct('name', name, 'solver', 'Scanning_Beam_Solver', ...
    'cfg', params, 'outputDir', outputDir);
end


% =========================================================================
function out = sanitize(v)
%SANITIZE  Make a value safe for save -v7 and jsonencode.
if isstruct(v)
    if numel(v) == 1
        out = struct();
        fn = fieldnames(v);
        for k = 1:numel(fn)
            out.(fn{k}) = sanitize(v.(fn{k}));
        end
    else
        for idx = numel(v):-1:1
            tmp(idx) = sanitize(v(idx));
        end
        out = reshape(tmp, size(v));
    end
elseif iscell(v)
    out = cellfun(@sanitize, v, 'UniformOutput', false);
elseif isa(v, 'function_handle')
    out = ['<function_handle> ' func2str(v)];
elseif isobject(v)
    out = ['<object:' class(v) '>'];
else
    out = v;
end
end


function writeJson(path, s)
txt = jsonencode(s, 'PrettyPrint', true);
fid = fopen(path, 'w');
assert(fid > 0, 'Cannot open %s for writing', path);
fwrite(fid, txt);
fclose(fid);
end
