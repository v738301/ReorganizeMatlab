function h = shadedErrorBar(x, y, err, varargin)
% SHADEDERRORBAR: Simple implementation for plotting line with shaded error bars
%
% Usage:
%   shadedErrorBar(x, y, err)
%   shadedErrorBar(x, y, err, 'lineprops', {'Color', [0.8 0.3 0.3], 'LineWidth', 2})
%
% Inputs:
%   x:   [1 × N] x-axis values
%   y:   [1 × N] mean values
%   err: [1 × N] error values (±1 SEM or SD)
%   varargin: Optional 'lineprops' followed by cell array of line properties
%
% Output:
%   h: struct with handles to patch and line objects

    % Parse inputs
    lineProps = {'Color', 'k', 'LineWidth', 1.5};

    if nargin > 3
        for i = 1:2:length(varargin)
            if strcmpi(varargin{i}, 'lineprops')
                lineProps = varargin{i+1};
            end
        end
    end

    % Ensure row vectors
    if size(x, 1) > 1
        x = x';
    end
    if size(y, 1) > 1
        y = y';
    end
    if size(err, 1) > 1
        err = err';
    end

    % Extract color from lineProps
    colorIdx = find(strcmpi(lineProps, 'Color'));
    if ~isempty(colorIdx) && colorIdx < length(lineProps)
        edgeColor = lineProps{colorIdx + 1};
    else
        edgeColor = 'k';
    end

    % Make patch lighter for shading (add transparency)
    if isnumeric(edgeColor)
        faceColor = min(edgeColor + 0.3, 1);  % Lighter version
        faceAlpha = 0.3;
    else
        faceColor = edgeColor;
        faceAlpha = 0.2;
    end

    % Create patch for error region
    x_patch = [x, fliplr(x)];
    y_patch = [y + err, fliplr(y - err)];

    % Remove NaN values
    valid = ~isnan(y_patch);
    x_patch = x_patch(valid);
    y_patch = y_patch(valid);

    % Plot shaded region
    h.patch = patch(x_patch, y_patch, faceColor, ...
        'EdgeColor', 'none', ...
        'FaceAlpha', faceAlpha);

    hold on;

    % Plot mean line
    h.mainLine = plot(x, y, lineProps{:});

    % Return to previous hold state
    % hold off is not called to allow additional plotting
end
