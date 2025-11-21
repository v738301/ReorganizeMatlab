% Inspect unit_features_for_clustering.mat structure
load('unit_features_for_clustering.mat');

fprintf('\n=== INSPECTING RESULTS STRUCTURE ===\n');
fprintf('Class: %s\n', class(results));
fprintf('Size: [%d x %d]\n', size(results, 1), size(results, 2));

if istable(results)
    fprintf('\nColumn names:\n');
    disp(results.Properties.VariableNames');
elseif isstruct(results)
    fprintf('\nField names:\n');
    disp(fieldnames(results));
else
    fprintf('\nResults content:\n');
    disp(results);
end
