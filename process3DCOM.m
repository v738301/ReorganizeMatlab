function [AdjustedXYZ, AdjustedXYZ_speed, processingInfo] = process3DCOM(Com_denoised, Fnew, varargin)
% PROCESS3DCOM - Process 3D center-of-mass trajectory data with boundary detection and filtering
%
% SYNOPSIS:
%   [AdjustedXYZ, AdjustedXYZ_speed, processingInfo] = process3DCOM(Com_denoised, Fnew)
%   [AdjustedXYZ, AdjustedXYZ_speed, processingInfo] = process3DCOM(Com_denoised, Fnew, 'param', value, ...)
%
% DESCRIPTION:
%   Processes 3D COM trajectory data to remove outliers, detect movement boundaries,
%   and produce clean, centered position data with automatic ground level detection.
%
% INPUTS:
%   Com_denoised  - Nx3 array of [X,Y,Z] COM positions (denoised)
%   Fnew          - Sampling frequency (Hz)
%
% OPTIONAL PARAMETERS (name-value pairs):
%   'maxRange'      - [1x3] Maximum allowed range for [X,Y,Z] (default: [600,600,100])
%   'minRange'      - [1x3] Minimum required range for [X,Y,Z] (default: [400,400,20])
%   'windowSizes'   - [1xN] Window size multipliers (default: [250,100,50])
%   'maxSpeed'      - Maximum physiological speed in units/s (default: 1000)
%   'maxZ'          - Maximum height threshold (default: 300)
%   'posThreshold'  - Max deviation from smoothed trajectory (default: 100)
%   'smoothWindow'  - Smoothing window in seconds (default: 1)
%   'zWindowSize'   - Z baseline window in seconds (default: 100)
%   'showPlots'     - Show visualization plots (default: true)
%   'verbose'       - Print processing information (default: true)
%
% OUTPUTS:
%   AdjustedXYZ     - Nx3 array of cleaned, centered, interpolated positions
%   AdjustedXYZ_speed - Nx1 array of instantaneous speeds (units/s)
%   processingInfo  - Struct containing:
%                     .boundaries - Detected boundaries for each dimension
%                     .midpoints - Midpoints used for centering
%                     .zeroZ - Detected ground level for Z-axis
%                     .outlierCount - Number of outliers removed
%                     .invalidCount - Number of invalid frames filtered
%                     .windowUsed - Window size that produced valid boundaries
%
% EXAMPLE:
%   % Basic usage with default parameters
%   [cleanPos, speed, info] = process3DCOM(rawData, 50);
%
%   % Custom parameters for smaller animal
%   [cleanPos, speed, info] = process3DCOM(rawData, 50, ...
%       'maxRange', [400,400,80], 'maxSpeed', 500, 'showPlots', false);
%
% Author: Optimized COM Processing
% Version: 1.0
% Date: 2025

%% Parse input parameters
p = inputParser;
addRequired(p, 'Com_denoised', @(x) isnumeric(x) && size(x,2)==3);
addRequired(p, 'Fnew', @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'maxRange', [600, 600, 200], @(x) isnumeric(x) && length(x)==3);
addParameter(p, 'minRange', [400, 400, 20], @(x) isnumeric(x) && length(x)==3);
addParameter(p, 'windowSizes', [250, 100, 50], @(x) isnumeric(x) && isvector(x));
addParameter(p, 'maxSpeed', 750, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'maxZ', 300, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'posThreshold', 100, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'smoothWindow', 1, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'zWindowSize', 100, @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'showPlots', true, @islogical);
addParameter(p, 'verbose', true, @islogical);

parse(p, Com_denoised, Fnew, varargin{:});
cfg = p.Results;

% Initialize
dims = {'X', 'Y', 'Z'};
colors = {[0 0.5 1], [0 0.7 0.4], [0.7 0 0.7]};
validBoundaries = false;
windowSizes = cfg.windowSizes * Fnew;

if cfg.verbose
    fprintf('\n=== STARTING 3D COM PROCESSING ===\n');
    fprintf('Data size: %d frames x 3 dimensions\n', size(Com_denoised, 1));
    fprintf('Sampling rate: %.1f Hz\n', Fnew);
end

%% BOUNDARY DETECTION
if cfg.showPlots
    fig1 = figure('Name', 'Boundary Detection', 'Position', [100, 100, 1400, 900]);
end

for wi = 1:length(windowSizes)
    ws = windowSizes(wi);
    
    % Calculate boundaries
    upper = movmax(Com_denoised, ws, 1);
    lower = movmin(Com_denoised, ws, 1);
    ranges = upper - lower;
    rejection = ranges > cfg.maxRange;
    
    % Check each dimension
    validDims = true(3,1);
    MM_temp = zeros(3,2);
    
    for d = 1:3
        validIdx = ~rejection(:,d);
        if sum(validIdx) > 0
            MM_temp(d,:) = [min(Com_denoised(validIdx,d)), max(Com_denoised(validIdx,d))];
            rangeWidth = MM_temp(d,2) - MM_temp(d,1);
            validDims(d) = rangeWidth >= cfg.minRange(d);
            
            if cfg.showPlots
                subplot(3, length(windowSizes)+1, (d-1)*(length(windowSizes)+1) + wi + 1);
                hold on;
                plot(Com_denoised(:,d), 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
                plot(find(validIdx), Com_denoised(validIdx,d), '.', 'MarkerSize', 3, 'Color', colors{d});
                plot(upper(:,d), '--', 'Color', [1 0.5 0], 'LineWidth', 1);
                plot(lower(:,d), '--', 'Color', [1 0.5 0], 'LineWidth', 1);
                yline(MM_temp(d,1), 'r-', 'Min', 'LineWidth', 1.5);
                yline(MM_temp(d,2), 'r-', 'Max', 'LineWidth', 1.5);
                title(sprintf('%s: Window=%ds, Range=%.1f', dims{d}, ws/Fnew, rangeWidth));
                ylabel(sprintf('%s Position', dims{d}));
                grid on;
            end
        else
            validDims(d) = false;
        end
    end
    
    % Store first valid solution
    if ~validBoundaries && all(validDims)
        validBoundaries = true;
        final.MM = MM_temp;
        final.mid = mean(MM_temp, 2);
        final.rejection = rejection;
        final.windowSize = ws;
        
        % Calculate Z true zero
        validZ = Com_denoised(~rejection(:,3), 3);
        lowerPercentile = validZ(validZ <= prctile(validZ, 25));
        [counts, edges] = histcounts(lowerPercentile, 50);
        [~, maxIdx] = max(counts);
        final.Zzero = (edges(maxIdx) + edges(maxIdx+1)) / 2;
        
        if cfg.verbose
            fprintf('Valid boundaries found with window size: %d samples (%.2f seconds)\n', ws, ws/Fnew);
            fprintf('X:[%.2f,%.2f] Y:[%.2f,%.2f] Z:[%.2f,%.2f] Zzero:%.2f\n', ...
                final.MM(1,:), final.MM(2,:), final.MM(3,:), final.Zzero);
        end
    end
end

% Add original data plots
if cfg.showPlots
    for d = 1:3
        subplot(3, length(windowSizes)+1, (d-1)*(length(windowSizes)+1) + 1);
        plot(Com_denoised(:,d), 'Color', [0.2 0.2 0.2], 'LineWidth', 1.5);
        title(sprintf('Original %s Data', dims{d}));
        ylabel(sprintf('%s Position', dims{d}));
        grid on;
    end
end

if ~validBoundaries
    error('process3DCOM:NoBoundaries', 'Failed to find valid boundaries. Consider adjusting parameters.');
end

%% APPLY FILTERING
% Determine Z threshold
if sum(Com_denoised(:,3) < final.Zzero)/length(Com_denoised(:,3)) > 0.3
    zThreshold = nanmedian(Com_denoised(:,3)) - mad(Com_denoised(:,3));
    if cfg.verbose
        fprintf('Using robust Z threshold: %.2f\n', zThreshold);
    end
else
    zThreshold = final.Zzero - 10;
    if cfg.verbose
        fprintf('Using true zero threshold: %.2f\n', zThreshold);
    end
end

% Mark outliers
outframes = Com_denoised(:,1) < final.MM(1,1) | Com_denoised(:,1) > final.MM(1,2) | ...
            Com_denoised(:,2) < final.MM(2,1) | Com_denoised(:,2) > final.MM(2,2) | ...
            Com_denoised(:,3) < zThreshold | Com_denoised(:,3) > final.MM(3,2);

AdjustedXYZ = Com_denoised;
AdjustedXYZ(outframes,:) = NaN;

% Fill NaN values
AdjustedXYZ_filled = fillBoundaryNaN(AdjustedXYZ);

%% ADDITIONAL FILTERING
smoothWindow = round(Fnew * cfg.smoothWindow);
smoothedXYZ = movmean(AdjustedXYZ_filled, smoothWindow, 1);
diffFromSmooth = sqrt(sum((AdjustedXYZ_filled - smoothedXYZ).^2, 2));

% Calculate speed
speed = [0; sqrt(sum(diff(AdjustedXYZ_filled).^2, 2)) * Fnew];
speed = fillmissing(speed, 'nearest');

% Identify invalid frames
invalidFrames = diffFromSmooth > cfg.posThreshold | ...
                AdjustedXYZ_filled(:,3) > cfg.maxZ;

% Expand invalidation window around high-speed points (1 second before and after)
speedInvalid = speed > cfg.maxSpeed;
if any(speedInvalid)
    expansionWindow = round(Fnew * 0.5); % 0.5 seconds on each side = 1 second total
    speedInvalidExpanded = false(size(speedInvalid));
    
    % Find all high-speed points and expand window around them
    highSpeedIdx = find(speedInvalid);
    for idx = highSpeedIdx'
        startIdx = max(1, idx - expansionWindow);
        endIdx = min(length(speedInvalid), idx + expansionWindow);
        speedInvalidExpanded(startIdx:endIdx) = true;
    end
    
    % Combine with other invalid frames
    invalidFrames = invalidFrames | speedInvalidExpanded;
    
    if cfg.verbose
        fprintf('Expanded %d high-speed points to %d invalid frames (±0.5s window)\n', ...
            sum(speedInvalid), sum(speedInvalidExpanded));
    end
end

AdjustedXYZ_filled(invalidFrames,:) = NaN;
AdjustedXYZ = fillmissing(AdjustedXYZ_filled, 'linear');

%% FINALIZE PROCESSING
% Center coordinates
AdjustedXYZ(:,1:2) = AdjustedXYZ(:,1:2) - final.mid(1:2)';

% Remove Z baseline drift
zWindowSize = min(Fnew * cfg.zWindowSize, floor(length(AdjustedXYZ)/10));
baseline = movmin(AdjustedXYZ(:,3), zWindowSize);
AdjustedXYZ(:,3) = AdjustedXYZ(:,3) - median(baseline);

% Calculate final speed
AdjustedXYZ_speed = [0; sqrt(sum(diff(AdjustedXYZ).^2, 2)) * Fnew];

%% VISUALIZATION
if cfg.showPlots
    % Summary figure
    figure('Name', 'Processing Summary', 'Position', [150, 150, 1200, 600]);
    
    for d = 1:3
        subplot(2,3,d);
        hold on;
        plot(Com_denoised(:,d), 'Color', [0.8 0.8 0.8], 'LineWidth', 1);
        plot(AdjustedXYZ(:,d) + (d<3)*final.mid(d), 'b-', 'LineWidth', 1.5);
        title(sprintf('%s: Before/After', dims{d}));
        ylabel('Position');
        legend('Original', 'Processed', 'Location', 'best');
        grid on;
        
        subplot(2,3,d+3);
        plot(AdjustedXYZ(:,d), 'Color', colors{d}, 'LineWidth', 1.5);
        title(sprintf('Final %s (Centered)', dims{d}));
        ylabel('Position');
        xlabel('Frames');
        grid on;
    end
    
    % 3D Trajectory
    figure('Name', 'Final 3D Trajectory', 'Position', [250, 250, 900, 700]);
    scatter3(AdjustedXYZ(:,1), AdjustedXYZ(:,2), AdjustedXYZ(:,3), 15, AdjustedXYZ_speed, 'filled');
    hold on;
    plot3(AdjustedXYZ(:,1), AdjustedXYZ(:,2), AdjustedXYZ(:,3), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
    c = colorbar;
    c.Label.String = 'Speed (units/s)';
    colormap(jet);
    xlabel('X (centered)');
    ylabel('Y (centered)');
    zlabel('Z (baseline removed)');
    title('Adjusted COM 3D Trajectory');
    grid on;
    axis equal;
    view(45, 30);
end

%% PREPARE OUTPUT
processingInfo.boundaries = final.MM;
processingInfo.midpoints = final.mid;
processingInfo.zeroZ = final.Zzero;
processingInfo.outlierCount = sum(outframes);
processingInfo.invalidCount = sum(invalidFrames);
processingInfo.windowUsed = final.windowSize;

if cfg.verbose
    fprintf('\n=== PROCESSING COMPLETE ===\n');
    fprintf('Outliers removed: %d (%.2f%%)\n', processingInfo.outlierCount, ...
        100*processingInfo.outlierCount/numel(outframes));
    fprintf('Invalid frames: %d (%.2f%%)\n', processingInfo.invalidCount, ...
        100*processingInfo.invalidCount/numel(invalidFrames));
    fprintf('Final ranges: X[%.1f,%.1f] Y[%.1f,%.1f] Z[%.1f,%.1f]\n', ...
        min(AdjustedXYZ(:,1)), max(AdjustedXYZ(:,1)), ...
        min(AdjustedXYZ(:,2)), max(AdjustedXYZ(:,2)), ...
        min(AdjustedXYZ(:,3)), max(AdjustedXYZ(:,3)));
    fprintf('===========================\n');
end

% Validation
if range(AdjustedXYZ(:,1)) > 650 || range(AdjustedXYZ(:,2)) > 650
    warning('process3DCOM:LargeRange', 'Range may be too large: X=%.2f, Y=%.2f', ...
        range(AdjustedXYZ(:,1)), range(AdjustedXYZ(:,2)));
end

end

%% HELPER FUNCTION
function data_filled = fillBoundaryNaN(data)
    % Fill NaN values with boundary extension and interpolation
    data_filled = data;
    
    for d = 1:size(data, 2)
        % Find valid indices
        validIdx = ~isnan(data(:,d));
        if sum(validIdx) < 2
            continue;
        end
        
        firstValid = find(validIdx, 1, 'first');
        lastValid = find(validIdx, 1, 'last');
        
        % Fill boundaries
        if firstValid > 1
            data_filled(1:firstValid-1, d) = data(firstValid, d);
        end
        if lastValid < size(data, 1)
            data_filled(lastValid+1:end, d) = data(lastValid, d);
        end
        
        % Interpolate middle
        nanIdx = isnan(data_filled(:,d));
        if any(nanIdx)
            validIndices = find(~nanIdx);
            if length(validIndices) > 1
                data_filled(nanIdx, d) = interp1(validIndices, ...
                    data_filled(validIndices, d), find(nanIdx), 'linear');
            end
        end
    end
end