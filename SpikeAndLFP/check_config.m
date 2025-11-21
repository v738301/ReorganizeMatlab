% Check configuration structure
load('unit_features_for_clustering.mat');

fprintf('\n=== CONFIGURATION ===\n');
fprintf('Config class: %s\n', class(results.config));

if isstruct(results.config)
    fprintf('\nConfig fields:\n');
    disp(results.config);
end
