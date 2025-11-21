% Quick verification of unit_features_for_clustering.mat
load('unit_features_for_clustering.mat');

fprintf('\n=== DATA VERIFICATION ===\n');
fprintf('Total units: %d\n', results.n_units);
fprintf('Total features: %d\n', results.n_features);
fprintf('Unique session count: %d\n', length(unique(results.master_features.Session)));

fprintf('\nUnique unit IDs (first 10):\n');
disp(results.master_features.UniqueUnitID(1:10));

fprintf('\nFeature columns (first 30):\n');
cols = results.master_features.Properties.VariableNames;
disp(cols(1:min(30,length(cols)))');

fprintf('\nSession types:\n');
disp(unique(results.master_features.SessionType));

fprintf('\nMissing data summary:\n');
missing_counts = sum(ismissing(results.master_features));
total_missing = sum(missing_counts);
total_cells = height(results.master_features) * width(results.master_features);
pct_missing = 100 * total_missing / total_cells;
fprintf('Columns with missing values: %d / %d\n', sum(missing_counts > 0), width(results.master_features));
fprintf('Total missing values: %d (%.2f%%)\n', total_missing, pct_missing);

fprintf('\nConfiguration:\n');
fprintf('  Low freq band: [%.1f, %.1f] Hz\n', results.config.low_freq_band);
fprintf('  High freq band: [%.1f, %.1f] Hz\n', results.config.high_freq_band);
fprintf('  PSTH window: [%.1f, %.1f] sec\n', results.config.psth_window);

fprintf('\n=== VERIFICATION COMPLETE ===\n');
