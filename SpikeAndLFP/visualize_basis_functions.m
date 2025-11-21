%% Visualize Gaussian Basis Functions
% Shows the 5 symmetric Gaussian basis functions used in the GLM

clear all
close all

% Parameters (same as in Test_GLM_Quick.m)
centers = [-0.5, 0, 0.5, 1.0, 1.5];  % Centers in seconds
width_fwhm = 1.0;                     % 1 second FWHM
bin_size = 0.05;                      % 50ms bins
n_bins = 60;                          % 3 seconds total (-1 to +2)

% Create basis functions
basis = createGaussianBasis(centers, width_fwhm, n_bins, bin_size);

% Time vector
time_vec = ((0:n_bins-1) * bin_size) - 1.0;  % -1.0 to +2.0 seconds
time_ms = time_vec * 1000;  % Convert to milliseconds

% Plot
figure('Position', [100, 100, 1200, 600]);

% Plot all basis functions
subplot(1, 2, 1);
colors = lines(5);
hold on;
for i = 1:5
    plot(time_ms, basis(:, i), 'LineWidth', 2.5, 'Color', colors(i,:), ...
         'DisplayName', sprintf('Basis %d (center=%.1fs)', i, centers(i)));
end

% Mark event onset
plot([0 0], ylim, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
text(0, max(basis(:))*0.9, ' Event', 'FontSize', 11, 'FontWeight', 'bold');

% Shade pre-event region
yl = ylim;
fill([-1000 0 0 -1000], [yl(1) yl(1) yl(2) yl(2)], ...
     [0.9 0.9 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
uistack(gca().Children(end), 'bottom');

xlabel('Time from Event (ms)', 'FontSize', 12);
ylabel('Basis Function Weight', 'FontSize', 12);
title('Gaussian Basis Functions (FWHM = 1.0s)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 10);
grid on;
xlim([-1000, 2000]);

% Plot stacked/filled
subplot(1, 2, 2);
area(time_ms, basis, 'LineWidth', 1.5);
colororder(colors);

% Mark event onset
hold on;
plot([0 0], [0 1], 'k--', 'LineWidth', 1.5);
text(0, 0.9, ' Event', 'FontSize', 11, 'FontWeight', 'bold');

xlabel('Time from Event (ms)', 'FontSize', 12);
ylabel('Cumulative Weight', 'FontSize', 12);
title('Stacked View (shows coverage)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
xlim([-1000, 2000]);
ylim([0, max(sum(basis, 2))]);

% Add legend
legend_labels = arrayfun(@(i) sprintf('Basis %d', i), 1:5, 'UniformOutput', false);
legend(legend_labels, 'Location', 'northeast', 'FontSize', 10);

% Save
saveas(gcf, 'Gaussian_Basis_Functions.png');
fprintf('✓ Saved Gaussian_Basis_Functions.png\n');

% Print basis function properties
fprintf('\nBasis Function Properties:\n');
fprintf('  FWHM: %.3f seconds (%.0f ms)\n', width_fwhm, width_fwhm*1000);
sigma = width_fwhm / (2 * sqrt(2 * log(2)));
fprintf('  Sigma: %.3f seconds (%.0f ms)\n', sigma, sigma*1000);
fprintf('  Centers: [');
fprintf('%.1f ', centers);
fprintf('] seconds\n');
fprintf('  Number of basis functions: %d\n', length(centers));
fprintf('  Temporal coverage: -1.0 to +2.0 seconds\n');
fprintf('\nEach basis function is normalized to sum to 1.0\n');


%% Helper function
function basis = createGaussianBasis(centers, width_fwhm, n_bins, bin_size)
    n_basis = length(centers);
    basis = zeros(n_bins, n_basis);
    sigma = width_fwhm / (2 * sqrt(2 * log(2)));
    time_vec = ((0:n_bins-1) * bin_size) - 1.0;

    for i = 1:n_basis
        center_time = centers(i);
        basis(:, i) = exp(-((time_vec - center_time).^2) / (2 * sigma^2));
        basis(:, i) = basis(:, i) / sum(basis(:, i));
    end
end
