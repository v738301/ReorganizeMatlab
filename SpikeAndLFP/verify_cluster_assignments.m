% Verify cluster assignments file
load('unit_cluster_assignments.mat');

fprintf('\n=== CLUSTER ASSIGNMENTS FILE VERIFICATION ===\n\n');

fprintf('File saved at: %s\n', cluster_assignments_output.timestamp);
fprintf('Cluster threshold: %.1f\n\n', cluster_assignments_output.config.cluster_threshold);

% Check Aversive sessions
if isfield(cluster_assignments_output, 'Aversive')
    fprintf('--- AVERSIVE SESSIONS ---\n');
    aversive_data = cluster_assignments_output.Aversive;
    fprintf('Total units: %d\n', height(aversive_data));
    fprintf('Variables: %s\n', strjoin(aversive_data.Properties.VariableNames, ', '));

    % Show cluster distribution
    cluster_counts = histcounts(aversive_data.ClusterID(~isnan(aversive_data.ClusterID)), ...
        1:(max(aversive_data.ClusterID)+1));
    fprintf('\nCluster distribution:\n');
    for c = 1:length(cluster_counts)
        if cluster_counts(c) > 0
            fprintf('  Cluster %d: %d units\n', c, cluster_counts(c));
        end
    end

    fprintf('\nFirst 10 units:\n');
    disp(aversive_data(1:min(10, height(aversive_data)), :));
end

% Check Reward sessions
if isfield(cluster_assignments_output, 'Reward')
    fprintf('\n--- REWARD SESSIONS ---\n');
    reward_data = cluster_assignments_output.Reward;
    fprintf('Total units: %d\n', height(reward_data));
    fprintf('Variables: %s\n', strjoin(reward_data.Properties.VariableNames, ', '));

    % Show cluster distribution
    cluster_counts = histcounts(reward_data.ClusterID(~isnan(reward_data.ClusterID)), ...
        1:(max(reward_data.ClusterID)+1));
    fprintf('\nCluster distribution:\n');
    for c = 1:length(cluster_counts)
        if cluster_counts(c) > 0
            fprintf('  Cluster %d: %d units\n', c, cluster_counts(c));
        end
    end

    fprintf('\nFirst 10 units:\n');
    disp(reward_data(1:min(10, height(reward_data)), :));
end

fprintf('\n=== VERIFICATION COMPLETE ===\n');
