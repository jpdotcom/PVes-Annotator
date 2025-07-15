%% Feature extraction for windowed data (non-real-time) - Enhanced Multi-Label Version
% Author: Vikram Abbaraju (vxa112@case.edu)
% Modified to handle both edited_dataset_A and edited_dataset_B
% Automatically detects and processes both datasets with different column names
% Creates individual dataset files AND combined files

%% Load and process both datasets
clear all;
close all;

% Define dataset files
dataset_files = {'extracted_trial_data_A.mat', 'extracted_trial_data_B.mat','extracted_trial_data_C.mat'};

% Create empty tables to store combined training data
training_data = table;

% Create cell array to store individual dataset features
dataset_features = cell(length(dataset_files), 1);
dataset_names = cell(length(dataset_files), 1);

%% Configuration parameters
INTERVAL_SIZE = 8;
Fc = 0.2;
MAX_SAMPLES_PER_CLASS = 4000; % Set this to control class balance

% Statistics for label overlap analysis
total_windows = 0;
single_label_windows = 0;
multi_label_windows = 0;
no_label_windows = 0;

%% Process each dataset
for dataset_idx = 1:length(dataset_files)
    dataset_file = dataset_files{dataset_idx};
    
    fprintf('\n=== Processing Dataset %s ===\n', dataset_file);
    
    % Check if file exists
    if ~exist(dataset_file, 'file')
        fprintf('Warning: %s not found. Skipping.\n', dataset_file);
        continue;
    end
    
    % Load the dataset
    fprintf('Loading: %s\n', dataset_file);
    S = load(dataset_file);
    
    % Get the trial data
    trial_data = S.trial_data;
    numTrials = S.numTrials;
    
    fprintf('Loaded %d trials for feature extraction\n', numTrials);
    
    % Initialize dataset-specific feature storage
    dataset_training_data = table;
    
    %% Auto-detect column names for different datasets
    % Check first trial to determine column structure
    sample_trial = trial_data{1};
    column_names = sample_trial.Properties.VariableNames;
    
    fprintf('Available columns: %s\n', strjoin(column_names, ', '));
    
    % Define mapping for different datasets
    % Dataset A uses: P_ves, abd, void, do
    % Dataset B uses: Pves, ABD, DO, VOID
    
    % Initialize column mapping
    pressure_col = '';
    abd_col = '';
    void_col = '';
    do_col = '';
    
    % Auto-detect pressure column
    if any(strcmp(column_names, 'P_ves'))
        pressure_col = 'P_ves';
        fprintf('Detected Dataset A format - using P_ves for pressure\n');
    elseif any(strcmp(column_names, 'Pves'))
        pressure_col = 'Pves';
        fprintf('Detected Dataset B format - using Pves for pressure\n');
    else
        error('Could not find pressure column (P_ves or Pves) in %s', dataset_file);
    end
    
    % Auto-detect abd column
    if any(strcmp(column_names, 'abd'))
        abd_col = 'abd';
    elseif any(strcmp(column_names, 'ABD'))
        abd_col = 'ABD';
    else
        error('Could not find abd column (abd or ABD) in %s', dataset_file);
    end
    
    % Auto-detect void column
    if any(strcmp(column_names, 'void'))
        void_col = 'void';
    elseif any(strcmp(column_names, 'VOID'))
        void_col = 'VOID';
    else
        error('Could not find void column (void or VOID) in %s', dataset_file);
    end
    
    % Auto-detect do column
    if any(strcmp(column_names, 'do'))
        do_col = 'do';
    elseif any(strcmp(column_names, 'DO'))
        do_col = 'DO';
    else
        error('Could not find do column (do or DO) in %s', dataset_file);
    end
    
    fprintf('Column mapping for %s:\n', dataset_file);
    fprintf('  Pressure: %s\n', pressure_col);
    fprintf('  ABD: %s\n', abd_col);
    fprintf('  VOID: %s\n', void_col);
    fprintf('  DO: %s\n', do_col);
    
    %% Process trials in current dataset
    for k = 1:numTrials
        fprintf('Processing trial %d/%d from %s\n', k, numTrials, dataset_file);
        
        data_table = table;
        
        % Extract data from the trial table using detected column names
        trial_table = trial_data{k};
        pves = trial_table.(pressure_col);
        abd = trial_table.(abd_col);
        void = trial_table.(void_col);
        do = trial_table.(do_col);
        
        % Pad signal to fit reshaping (ensure divisible by INTERVAL_SIZE)
        x = mod(-mod(length(pves), INTERVAL_SIZE), INTERVAL_SIZE);
        pves = [pves' zeros(1, x)]';
        
        % Optional: Apply EMA filter (uncomment if needed)
        % pves = ema_filter(pves, Fc)';
        
        dwt_size = length(pves);
        
        % Process labels - pad and reshape into windows
        % Instead of simple sum, calculate label statistics per window
        abd_padded = [abd' zeros(1, x)]';
        void_padded = [void' zeros(1, x)]';
        do_padded = [do' zeros(1, x)]';
        
        % Reshape into windows
        abd_windows = reshape(abd_padded, INTERVAL_SIZE, length(abd_padded)/INTERVAL_SIZE);
        void_windows = reshape(void_padded, INTERVAL_SIZE, length(void_padded)/INTERVAL_SIZE);
        do_windows = reshape(do_padded, INTERVAL_SIZE, length(do_padded)/INTERVAL_SIZE);
        
        % Calculate label statistics for each window
        num_windows = size(abd_windows, 2);
        
        % Binary labels (original approach)
        abd_binary = sum(abd_windows, 1) >= 1;
        void_binary = sum(void_windows, 1) >= 1;
        do_binary = sum(do_windows, 1) >= 1;
        
        % Label proportions (what fraction of window has each label)
        abd_proportion = sum(abd_windows, 1) / INTERVAL_SIZE;
        void_proportion = sum(void_windows, 1) / INTERVAL_SIZE;
        do_proportion = sum(do_windows, 1) / INTERVAL_SIZE;
        
        % Label counts (how many samples in window have each label)
        abd_count = sum(abd_windows, 1);
        void_count = sum(void_windows, 1);
        do_count = sum(do_windows, 1);
        
        % Multi-label statistics
        total_labels_per_window = abd_binary + void_binary + do_binary;
        
        % Update statistics
        total_windows = total_windows + num_windows;
        single_label_windows = single_label_windows + sum(total_labels_per_window == 1);
        multi_label_windows = multi_label_windows + sum(total_labels_per_window > 1);
        no_label_windows = no_label_windows + sum(total_labels_per_window == 0);
        
        % Store all label representations (using standardized names)
        data_table.abd = abd_binary';
        data_table.void = void_binary';
        data_table.do = do_binary';
        
        % Additional multi-label features
        data_table.abd_proportion = abd_proportion';
        data_table.void_proportion = void_proportion';
        data_table.do_proportion = do_proportion';
        
        data_table.abd_count = abd_count';
        data_table.void_count = void_count';
        data_table.do_count = do_count';
        
        data_table.total_labels = total_labels_per_window';
        data_table.is_multi_label = (total_labels_per_window > 1)';
        
        % Add dataset identifier
        data_table.dataset = repmat(string(dataset_file), num_windows, 1);
        
        % Create time vector
        time = [0:0.1:(dwt_size - 1)*0.1];
        
        % Compute 5-level DWT
        [cA1, cD1] = dwt(pves, 'db2', 'mode', 'per');
        [cA2, cD2] = dwt(cA1, 'db2', 'mode', 'per');
        [cA3, cD3] = dwt(cA2, 'db2', 'mode', 'per');
        [cA4, cD4] = dwt(cA3, 'db2', 'mode', 'per');
        [cA5, cD5] = dwt(cA4, 'db2', 'mode', 'per');
        
        % Interpolate DWT components to original signal length
        cA1 = spline(downsample(time, 2^1), cA1, time);
        cD1 = spline(downsample(time, 2^1), cD1, time);
        cA2 = spline(downsample(time, 2^2), cA2, time);
        cD2 = spline(downsample(time, 2^2), cD2, time);
        cA3 = spline(downsample(time, 2^3), cA3, time);
        cD3 = spline(downsample(time, 2^3), cD3, time);
        cA4 = spline(downsample(time, 2^4), cA4, time);
        cD4 = spline(downsample(time, 2^4), cD4, time);
        cA5 = spline(downsample(time, 2^5), cA5, time);
        cD5 = spline(downsample(time, 2^5), cD5, time);
        
        % Extract features from each DWT level
        % Level 1
        cA1_matrix = reshape(cA1, INTERVAL_SIZE, dwt_size/INTERVAL_SIZE);
        cA1_max = max(cA1_matrix, [], 1);
        cA1_mav = mean(abs(cA1_matrix), 1);
        cA1_med = median(cA1_matrix, 1);
        cD1_matrix = reshape(cD1, INTERVAL_SIZE, dwt_size/INTERVAL_SIZE);
        cD1_max = max(cD1_matrix, [], 1);
        cD1_mav = mean(abs(cD1_matrix), 1);
        cD1_med = median(cD1_matrix, 1);
        
        % Level 2
        cA2_matrix = reshape(cA2, INTERVAL_SIZE, dwt_size/INTERVAL_SIZE);
        cA2_max = max(cA2_matrix, [], 1);
        cA2_mav = mean(abs(cA2_matrix), 1);
        cA2_med = median(cA2_matrix, 1);
        cD2_matrix = reshape(cD2, INTERVAL_SIZE, dwt_size/INTERVAL_SIZE);
        cD2_max = max(cD2_matrix, [], 1);
        cD2_mav = mean(abs(cD2_matrix), 1);
        cD2_med = median(cD2_matrix, 1);
        
        % Level 3
        cA3_matrix = reshape(cA3, INTERVAL_SIZE, dwt_size/INTERVAL_SIZE);
        cA3_max = max(cA3_matrix, [], 1);
        cA3_mav = mean(abs(cA3_matrix), 1);
        cA3_med = median(cA3_matrix, 1);
        cD3_matrix = reshape(cD3, INTERVAL_SIZE, dwt_size/INTERVAL_SIZE);
        cD3_max = max(cD3_matrix, [], 1);
        cD3_mav = mean(abs(cD3_matrix), 1);
        cD3_med = median(cD3_matrix, 1);
        
        % Level 4
        cA4_matrix = reshape(cA4, INTERVAL_SIZE, dwt_size/INTERVAL_SIZE);
        cA4_max = max(cA4_matrix, [], 1);
        cA4_mav = mean(abs(cA4_matrix), 1);
        cA4_med = median(cA4_matrix, 1);
        cD4_matrix = reshape(cD4, INTERVAL_SIZE, dwt_size/INTERVAL_SIZE);
        cD4_max = max(cD4_matrix, [], 1);
        cD4_mav = mean(abs(cD4_matrix), 1);
        cD4_med = median(cD4_matrix, 1);
        
        % Level 5
        cA5_matrix = reshape(cA5, INTERVAL_SIZE, dwt_size/INTERVAL_SIZE);
        cA5_max = max(cA5_matrix, [], 1);
        cA5_mav = mean(abs(cA5_matrix), 1);
        cA5_med = median(cA5_matrix, 1);
        cD5_matrix = reshape(cD5, INTERVAL_SIZE, dwt_size/INTERVAL_SIZE);
        cD5_max = max(cD5_matrix, [], 1);
        cD5_mav = mean(abs(cD5_matrix), 1);
        cD5_med = median(cD5_matrix, 1);
        
        % Cross-correlation features
        xcorr1_matrix = reshape([xcorr(cA1, cD1) 0], INTERVAL_SIZE*2, dwt_size/INTERVAL_SIZE);
        xcorr1_med = median(xcorr1_matrix, 1);
        xcorr1_mean = mean(xcorr1_matrix, 1);
        xcorr1_max = max(xcorr1_matrix, [], 1);
        
        xcorr2_matrix = reshape([xcorr(cA2, cD2) 0], INTERVAL_SIZE*2, dwt_size/INTERVAL_SIZE);
        xcorr2_med = median(xcorr2_matrix, 1);
        xcorr2_mean = mean(xcorr2_matrix, 1);
        xcorr2_max = max(xcorr2_matrix, [], 1);
        
        xcorr3_matrix = reshape([xcorr(cA3, cD3) 0], INTERVAL_SIZE*2, dwt_size/INTERVAL_SIZE);
        xcorr3_med = median(xcorr3_matrix, 1);
        xcorr3_mean = mean(xcorr3_matrix, 1);
        xcorr3_max = max(xcorr3_matrix, [], 1);
        
        xcorr4_matrix = reshape([xcorr(cA4, cD4) 0], INTERVAL_SIZE*2, dwt_size/INTERVAL_SIZE);
        xcorr4_med = median(xcorr4_matrix, 1);
        xcorr4_mean = mean(xcorr4_matrix, 1);
        xcorr4_max = max(xcorr4_matrix, [], 1);
        
        xcorr5_matrix = reshape([xcorr(cA5, cD5) 0], INTERVAL_SIZE*2, dwt_size/INTERVAL_SIZE);
        xcorr5_med = median(xcorr5_matrix, 1);
        xcorr5_mean = mean(xcorr5_matrix, 1);
        xcorr5_max = max(xcorr5_matrix, [], 1);
        
        % Entropy features
        cA1_ent = zeros(1, dwt_size/INTERVAL_SIZE);
        cD1_ent = zeros(1, dwt_size/INTERVAL_SIZE);
        cA2_ent = zeros(1, dwt_size/INTERVAL_SIZE);
        cD2_ent = zeros(1, dwt_size/INTERVAL_SIZE);
        cA3_ent = zeros(1, dwt_size/INTERVAL_SIZE);
        cD3_ent = zeros(1, dwt_size/INTERVAL_SIZE);
        cA4_ent = zeros(1, dwt_size/INTERVAL_SIZE);
        cD4_ent = zeros(1, dwt_size/INTERVAL_SIZE);
        cA5_ent = zeros(1, dwt_size/INTERVAL_SIZE);
        cD5_ent = zeros(1, dwt_size/INTERVAL_SIZE);
        
        for j = 1:dwt_size/INTERVAL_SIZE
            cA1_ent(j) = wentropy(cA1_matrix(:, j), 'shannon');
            cD1_ent(j) = wentropy(cD1_matrix(:, j), 'shannon');
            cA2_ent(j) = wentropy(cA2_matrix(:, j), 'shannon');
            cD2_ent(j) = wentropy(cD2_matrix(:, j), 'shannon');
            cA3_ent(j) = wentropy(cA3_matrix(:, j), 'shannon');
            cD3_ent(j) = wentropy(cD3_matrix(:, j), 'shannon');
            cA4_ent(j) = wentropy(cA4_matrix(:, j), 'shannon');
            cD4_ent(j) = wentropy(cD4_matrix(:, j), 'shannon');
            cA5_ent(j) = wentropy(cA5_matrix(:, j), 'shannon');
            cD5_ent(j) = wentropy(cD5_matrix(:, j), 'shannon');
        end
        
        % Add all features to data table
        data_table.cA1_max = cA1_max';
        data_table.cA1_mav = cA1_mav';
        data_table.cA1_med = cA1_med';
        data_table.cA1_ent = cA1_ent';
        data_table.cD1_max = cD1_max';
        data_table.cD1_mav = cD1_mav';
        data_table.cD1_med = cD1_med';
        data_table.cD1_ent = cD1_ent';
        data_table.xcorr1_med = xcorr1_med';
        data_table.xcorr1_mean = xcorr1_mean';
        data_table.xcorr1_max = xcorr1_max';
        
        data_table.cA2_max = cA2_max';
        data_table.cA2_mav = cA2_mav';
        data_table.cA2_med = cA2_med';
        data_table.cA2_ent = cA2_ent';
        data_table.cD2_max = cD2_max';
        data_table.cD2_mav = cD2_mav';
        data_table.cD2_med = cD2_med';
        data_table.cD2_ent = cD2_ent';
        data_table.xcorr2_med = xcorr2_med';
        data_table.xcorr2_mean = xcorr2_mean';
        data_table.xcorr2_max = xcorr2_max';
        
        data_table.cA3_max = cA3_max';
        data_table.cA3_mav = cA3_mav';
        data_table.cA3_med = cA3_med';
        data_table.cA3_ent = cA3_ent';
        data_table.cD3_max = cD3_max';
        data_table.cD3_mav = cD3_mav';
        data_table.cD3_med = cD3_med';
        data_table.cD3_ent = cD3_ent';
        data_table.xcorr3_med = xcorr3_med';
        data_table.xcorr3_mean = xcorr3_mean';
        data_table.xcorr3_max = xcorr3_max';
        
        data_table.cA4_max = cA4_max';
        data_table.cA4_mav = cA4_mav';
        data_table.cA4_med = cA4_med';
        data_table.cA4_ent = cA4_ent';
        data_table.cD4_max = cD4_max';
        data_table.cD4_mav = cD4_mav';
        data_table.cD4_med = cD4_med';
        data_table.cD4_ent = cD4_ent';
        data_table.xcorr4_med = xcorr4_med';
        data_table.xcorr4_mean = xcorr4_mean';
        data_table.xcorr4_max = xcorr4_max';
        
        data_table.cA5_max = cA5_max';
        data_table.cA5_mav = cA5_mav';
        data_table.cA5_med = cA5_med';
        data_table.cA5_ent = cA5_ent';
        data_table.cD5_max = cD5_max';
        data_table.cD5_mav = cD5_mav';
        data_table.cD5_med = cD5_med';
        data_table.cD5_ent = cD5_ent';
        data_table.xcorr5_med = xcorr5_med';
        data_table.xcorr5_mean = xcorr5_mean';
        data_table.xcorr5_max = xcorr5_max';
        
        % Add to dataset-specific training data
        dataset_training_data = [dataset_training_data; data_table];
        
        % Combine with overall training data
        training_data = [training_data; data_table];
    end
    
    % Store dataset features and name
    dataset_features{dataset_idx} = dataset_training_data;
    dataset_names{dataset_idx} = dataset_file;
    
    % Save individual dataset features
    dataset_name = strrep(dataset_file, '.mat', '');
    dataset_name = strrep(dataset_name, 'extracted_trial_data_', '');
    
    % Save individual dataset files with multiple formats
    individual_training_data = dataset_training_data;
    
    % Multi-label format
    save(sprintf('training_data_multi_label_%s.mat', dataset_name), 'individual_training_data');
    fprintf('Saved: training_data_multi_label_%s.mat (%d samples)\n', dataset_name, size(individual_training_data, 1));
    
    % Single class format (remove multi-label specific columns)
    individual_training_data_single = removevars(individual_training_data, ...
        {'abd_proportion', 'void_proportion', 'do_proportion', ...
         'abd_count', 'void_count', 'do_count', 'total_labels', 'is_multi_label', 'dataset'});
    save(sprintf('training_data_single_class_%s.mat', dataset_name), 'individual_training_data_single');
    fprintf('Saved: training_data_single_class_%s.mat (%d samples)\n', dataset_name, size(individual_training_data_single, 1));
    
    % Multi-class format (with priority handling)
    event = "none" + strings(size(individual_training_data_single, 1), 1);
    event(individual_training_data_single.do == 1) = "do";
    event(individual_training_data_single.abd == 1) = "abd";
    event(individual_training_data_single.void == 1) = "void";  % Highest priority
    individual_training_data_multi_class = individual_training_data_single;
    individual_training_data_multi_class.class = event;
    individual_training_data_multi_class = removevars(individual_training_data_multi_class, {'do', 'abd', 'void'});
    save(sprintf('training_data_multi_class_%s.mat', dataset_name), 'individual_training_data_multi_class');
    fprintf('Saved: training_data_multi_class_%s.mat (%d samples)\n', dataset_name, size(individual_training_data_multi_class, 1));
    
    fprintf('Completed processing %s\n', dataset_file);
end

fprintf('\nFeature extraction complete. Total samples: %d\n', size(training_data, 1));

% Print multi-label statistics
fprintf('\nCombined Multi-label Statistics:\n');
fprintf('Total windows: %d\n', total_windows);
fprintf('No label windows: %d (%.2f%%)\n', no_label_windows, 100*no_label_windows/total_windows);
fprintf('Single label windows: %d (%.2f%%)\n', single_label_windows, 100*single_label_windows/total_windows);
fprintf('Multi-label windows: %d (%.2f%%)\n', multi_label_windows, 100*multi_label_windows/total_windows);

% Print dataset distribution
fprintf('\nDataset Distribution:\n');
for i = 1:length(dataset_files)
    dataset_name = dataset_files{i};
    count = sum(strcmp(training_data.dataset, dataset_name));
    fprintf('%s: %d samples (%.2f%%)\n', dataset_name, count, 100*count/size(training_data, 1));
end

% Print individual dataset statistics
fprintf('\nIndividual Dataset Statistics:\n');
for i = 1:length(dataset_features)
    if ~isempty(dataset_features{i})
        dataset_data = dataset_features{i};
        fprintf('\n%s:\n', dataset_names{i});
        fprintf('  Total samples: %d\n', size(dataset_data, 1));
        
        % Label distribution
        abd_count = sum(dataset_data.abd);
        void_count = sum(dataset_data.void);
        do_count = sum(dataset_data.do);
        none_count = sum(dataset_data.abd == 0 & dataset_data.void == 0 & dataset_data.do == 0);
        multi_label_count = sum(dataset_data.is_multi_label);
        
        fprintf('  Label distribution:\n');
        fprintf('    Abd: %d (%.2f%%)\n', abd_count, 100*abd_count/size(dataset_data, 1));
        fprintf('    Void: %d (%.2f%%)\n', void_count, 100*void_count/size(dataset_data, 1));
        fprintf('    Do: %d (%.2f%%)\n', do_count, 100*do_count/size(dataset_data, 1));
        fprintf('    None: %d (%.2f%%)\n', none_count, 100*none_count/size(dataset_data, 1));
        fprintf('    Multi-label: %d (%.2f%%)\n', multi_label_count, 100*multi_label_count/size(dataset_data, 1));
    end
end

%% Save combined training data

% Training data for multi-label classification (preserves all label information)
training_data_multi_label = training_data;
save training_data_multi_label_combined.mat training_data_multi_label;
fprintf('\nSaved: training_data_multi_label_combined.mat\n');

% Training data for binary classification for each class (original approach)
training_data_single_class = training_data;
% Remove the additional multi-label features and dataset identifier for compatibility
training_data_single_class = removevars(training_data_single_class, ...
    {'abd_proportion', 'void_proportion', 'do_proportion', ...
     'abd_count', 'void_count', 'do_count', 'total_labels', 'is_multi_label', 'dataset'});
save training_data_single_class_combined.mat training_data_single_class;
fprintf('Saved: training_data_single_class_combined.mat\n');

% Training data for multi-class classification (handles overlapping labels with priority)
% Priority order: void > abd > do > none
event = "none" + strings(size(training_data_single_class, 1), 1);
event(training_data_single_class.do == 1) = "do";
event(training_data_single_class.abd == 1) = "abd";
event(training_data_single_class.void == 1) = "void";  % Highest priority
training_data_multi_class = training_data_single_class;
training_data_multi_class.class = event;
training_data_multi_class = removevars(training_data_multi_class, {'do', 'abd', 'void'});
save training_data_multi_class_combined.mat training_data_multi_class;
fprintf('Saved: training_data_multi_class_combined.mat\n');

% Training data for multi-class classification with multi-label indicator
training_data_multi_class_enhanced = training_data_multi_class;
training_data_multi_class_enhanced.is_multi_label = training_data.is_multi_label;
training_data_multi_class_enhanced.dataset = training_data.dataset;
save training_data_multi_class_enhanced_combined.mat training_data_multi_class_enhanced;
fprintf('Saved: training_data_multi_class_enhanced_combined.mat\n');

% Training data for binary classification between event/no event
training_data_coarse = training_data_multi_class;
event = "none" + strings(size(training_data_multi_class, 1), 1);
event(~strcmp(training_data_multi_class.class, "none")) = "event";
training_data_coarse.class = event;
save training_data_coarse_combined.mat training_data_coarse;
fprintf('Saved: training_data_coarse_combined.mat\n');

%% Create balanced datasets with configurable class sizes

% For multi-class classification (balanced)
none_indices = find(strcmp(training_data_multi_class.class, "none"));
void_indices = find(strcmp(training_data_multi_class.class, "void"));
abd_indices = find(strcmp(training_data_multi_class.class, "abd"));
do_indices = find(strcmp(training_data_multi_class.class, "do"));

fprintf('\nOriginal class distribution:\n');
fprintf('  None: %d\n', length(none_indices));
fprintf('  Void: %d\n', length(void_indices));
fprintf('  Abd: %d\n', length(abd_indices));
fprintf('  Do: %d\n', length(do_indices));

% Randomly sample up to MAX_SAMPLES_PER_CLASS from each class
none_indices = none_indices(randperm(length(none_indices)));
none_indices = none_indices(1:min(MAX_SAMPLES_PER_CLASS, length(none_indices)));

void_indices = void_indices(randperm(length(void_indices)));
void_indices = void_indices(1:min(MAX_SAMPLES_PER_CLASS, length(void_indices)));

abd_indices = abd_indices(randperm(length(abd_indices)));
abd_indices = abd_indices(1:min(MAX_SAMPLES_PER_CLASS, length(abd_indices)));

do_indices = do_indices(randperm(length(do_indices)));
do_indices = do_indices(1:min(MAX_SAMPLES_PER_CLASS, length(do_indices)));

fprintf('\nBalanced class distribution (max %d per class):\n', MAX_SAMPLES_PER_CLASS);
fprintf('  None: %d\n', length(none_indices));
fprintf('  Void: %d\n', length(void_indices));
fprintf('  Abd: %d\n', length(abd_indices));
fprintf('  Do: %d\n', length(do_indices));

% Combine balanced dataset
training_data_fine = [training_data_multi_class(none_indices, :); 
                      training_data_multi_class(void_indices, :); 
                      training_data_multi_class(do_indices, :); 
                      training_data_multi_class(abd_indices, :)];

% Shuffle the training data
training_data_fine = training_data_fine(randperm(size(training_data_fine, 1)), :);

save training_data_fine_combined.mat training_data_fine;
fprintf('Saved: training_data_fine_combined.mat (balanced dataset with %d samples)\n', size(training_data_fine, 1));

%% Create balanced multi-label dataset with equal label counts
% Goal: Have roughly equal counts of abd, void, do, and none labels in final dataset

% Count current label occurrences
abd_count = sum(training_data.abd);
void_count = sum(training_data.void);
do_count = sum(training_data.do);
none_count = sum(training_data.abd == 0 & training_data.void == 0 & training_data.do == 0);

fprintf('\nOriginal label counts:\n');
fprintf('  Abd: %d\n', abd_count);
fprintf('  Void: %d\n', void_count);
fprintf('  Do: %d\n', do_count);
fprintf('  None: %d\n', none_count);

% Target count for each label (use the minimum available or MAX_SAMPLES_PER_CLASS)
target_count = min([abd_count, void_count, do_count, none_count, MAX_SAMPLES_PER_CLASS]);

fprintf('\nTarget count for each label: %d\n', target_count);

% Create balanced dataset by iterative sampling
selected_indices = [];
current_abd_count = 0;
current_void_count = 0;
current_do_count = 0;
current_none_count = 0;

% Shuffle all indices to ensure random sampling
all_indices = randperm(size(training_data, 1));

for i = all_indices
    % Check current sample's labels
    has_abd = training_data.abd(i) == 1;
    has_void = training_data.void(i) == 1;
    has_do = training_data.do(i) == 1;
    has_none = ~has_abd & ~has_void & ~has_do;
    
    % Check if we still need any of these labels
    need_abd = has_abd && current_abd_count < target_count;
    need_void = has_void && current_void_count < target_count;
    need_do = has_do && current_do_count < target_count;
    need_none = has_none && current_none_count < target_count;
    
    % Add sample if we need any of its labels
    if need_abd || need_void || need_do || need_none
        selected_indices = [selected_indices; i];
        
        % Update counts
        if has_abd
            current_abd_count = current_abd_count + 1;
        end
        if has_void
            current_void_count = current_void_count + 1;
        end
        if has_do
            current_do_count = current_do_count + 1;
        end
        if has_none
            current_none_count = current_none_count + 1;
        end
    end
    
    % Stop if we've reached target for all labels
    if current_abd_count >= target_count && current_void_count >= target_count && ...
       current_do_count >= target_count && current_none_count >= target_count
        break;
    end
end

% Create balanced multi-label dataset
training_data_multi_label_balanced = training_data(selected_indices, :);

% Shuffle the final dataset
training_data_multi_label_balanced = training_data_multi_label_balanced(randperm(size(training_data_multi_label_balanced, 1)), :);

save training_data_multi_label_balanced.mat training_data_multi_label_balanced;
fprintf('Saved: training_data_multi_label_balanced.mat (%d samples)\n', size(training_data_multi_label_balanced, 1));

% Print final balanced distribution
final_abd_count = sum(training_data_multi_label_balanced.abd);
final_void_count = sum(training_data_multi_label_balanced.void);
final_do_count = sum(training_data_multi_label_balanced.do);
final_none_count = sum(training_data_multi_label_balanced.abd == 0 & ...
                       training_data_multi_label_balanced.void == 0 & ...
                       training_data_multi_label_balanced.do == 0);

fprintf('\nFinal balanced label counts:\n');
fprintf('  Abd: %d\n', final_abd_count);
fprintf('  Void: %d\n', final_void_count);
fprintf('  Do: %d\n', final_do_count);
fprintf('  None: %d\n', final_none_count);
fprintf('  Multi-label samples: %d\n', sum(training_data_multi_label_balanced.is_multi_label));

% Calculate balance ratio (how close to perfectly balanced)
counts = [final_abd_count, final_void_count, final_do_count, final_none_count];
balance_ratio = min(counts) / max(counts);
fprintf('  Balance ratio: %.3f (1.0 = perfectly balanced)\n', balance_ratio);

% Training data specifically for multi-label learning (only windows with multiple labels)
multi_label_indices = find(training_data.is_multi_label == 1);
if ~isempty(multi_label_indices)
    training_data_multi_label_only = training_data(multi_label_indices, :);
    save training_data_multi_label_only.mat training_data_multi_label_only;
    fprintf('Saved: training_data_multi_label_only.mat (%d multi-label samples)\n', length(multi_label_indices));
else
    fprintf('No multi-label samples found.\n');
end

fprintf('\nAll training datasets created successfully!\n');
fprintf('New datasets include enhanced multi-label information and statistics.\n');