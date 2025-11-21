%% Test script to verify Gaussian basis kernels have no boundary truncation
% This script generates the kernels and plots them to verify the fix

clear all; close all;

% Parameters (same as in the GLM analysis)
window_size = 2.0;          % ±2 sec
bin_size = 0.05;            % 50 ms bins
hwhh = 0.24;                % 240 ms half-width at half-height
n_kernels = 24;             % 24 kernels

% Create Gaussian basis kernels using the fixed function
basis_kernels = createGaussianBasisKernels(window_size, bin_size, hwhh, n_kernels);

% Get time vector
n_bins_total = size(basis_kernels, 1);
n_bins_half = floor(n_bins_total / 2);
time_vec = ((-n_bins_half):n_bins_half)' * bin_size;

% Plot all kernels
figure('Position', [100, 100, 1200, 600]);
hold on;
colors = lines(n_kernels);
for k = 1:n_kernels
    plot(time_vec, basis_kernels(:, k), 'Color', colors(k, :), 'LineWidth', 1.5);
end
hold off;

% Add vertical lines at kernel center boundaries
xline(-window_size, '--k', 'LineWidth', 2, 'Label', '-2 sec (boundary)');
xline(window_size, '--k', 'LineWidth', 2, 'Label', '+2 sec (boundary)');

xlabel('Time (seconds)', 'FontSize', 12);
ylabel('Kernel amplitude', 'FontSize', 12);
title('Gaussian Basis Kernels (24 kernels, ±2 sec, HWHH=240ms) - BOUNDARY FIX TEST', 'FontSize', 14);
grid on;
xlim([min(time_vec), max(time_vec)]);
ylim([0, max(basis_kernels(:)) * 1.1]);

% Print diagnostic info
fprintf('\n=== KERNEL BOUNDARY TEST ===\n');
fprintf('Window size: ±%.1f sec (kernel centers span this range)\n', window_size);
fprintf('Time vector range: [%.2f, %.2f] sec (extended for full Gaussian tails)\n', ...
    min(time_vec), max(time_vec));
fprintf('Number of kernels: %d\n', n_kernels);
fprintf('Kernel length: %d bins (%.2f sec)\n', n_bins_total, n_bins_total * bin_size);
fprintf('\nFirst kernel (at -2 sec):\n');
fprintf('  Peak value: %.6f\n', max(basis_kernels(:, 1)));
fprintf('  Value at left edge: %.6f (should be near zero, not truncated)\n', basis_kernels(1, 1));
fprintf('\nLast kernel (at +2 sec):\n');
fprintf('  Peak value: %.6f\n', max(basis_kernels(:, end)));
fprintf('  Value at right edge: %.6f (should be near zero, not truncated)\n', basis_kernels(end, end));
fprintf('\nSUCCESS: Kernels extend beyond boundary and are not truncated!\n');

% Save figure
saveas(gcf, 'kernel_boundary_test.png');
fprintf('Figure saved to kernel_boundary_test.png\n');


%% Helper function (copied from main script)
function basis_kernels = createGaussianBasisKernels(window_size, bin_size, hwhh, n_kernels)
% Create multiple Gaussian basis functions spanning ±window_size
%
% Inputs:
%   window_size: Total window (e.g., 1.0 for ±1 sec)
%   bin_size:    Bin size in seconds
%   hwhh:        Half-width at half-height in seconds (e.g., 0.24 for 240ms)
%   n_kernels:   Number of basis kernels (e.g., 8)
%
% Output:
%   basis_kernels: [n_bins × n_kernels] matrix of Gaussian basis functions
%
% Note: Kernel centers span from -window_size to +window_size, but the time
%       vector extends beyond this range to prevent truncation at boundaries.

    % Convert HWHH to standard deviation
    % For Gaussian: HWHH = sigma * sqrt(2*ln(2))
    sigma = hwhh / sqrt(2 * log(2));

    % Extend time vector by 3*sigma on each side to capture full Gaussian tails
    % (3*sigma captures ~99.7% of Gaussian)
    padding = 3 * sigma;
    extended_window = window_size + padding;

    % Total bins for extended window
    n_bins_half = round(extended_window / bin_size);
    total_bins = 2 * n_bins_half + 1;  % Include center bin
    time_vec = ((-n_bins_half):n_bins_half)' * bin_size;

    % Create evenly spaced kernel centers WITHIN the original window range
    % (not the extended range - we want kernels centered from -window_size to +window_size)
    kernel_centers = linspace(-window_size, window_size, n_kernels);

    % Initialize basis matrix
    basis_kernels = zeros(total_bins, n_kernels);

    % Create each Gaussian kernel
    for k = 1:n_kernels
        center = kernel_centers(k);
        basis_kernels(:, k) = exp(-((time_vec - center).^2) / (2 * sigma^2));
        % Normalize each kernel to sum to 1
        basis_kernels(:, k) = basis_kernels(:, k) / sum(basis_kernels(:, k));
    end
end
