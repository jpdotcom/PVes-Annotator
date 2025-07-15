function extract_annotated_data_universal()
% Extract pressure and annotation data from edited datasets A, B, or C
% Dataset A: Uses fixed column indices
% Dataset B/C: Use column names
% Modified to exclude invalid annotations from the extracted data

% === Load the edited dataset ===
[file, path] = uigetfile('*.mat', 'Select edited dataset file (A, B, or C)');
if isequal(file, 0)
    fprintf('No file selected. Exiting.\n');
    return;
end

fprintf('Loading: %s\n', fullfile(path, file));
S = load(fullfile(path, file));

% Try to find the dataset variable (handle different possible names)
dataset = [];
dataset_type = '';

if isfield(S, 'dataset_A_annotated')
    dataset = S.dataset_A_annotated;
    dataset_type = 'A';
elseif isfield(S, 'dataset_B_annotated')
    dataset = S.dataset_B_annotated;
    dataset_type = 'B';
elseif isfield(S, 'dataset_C_annotated')
    dataset = S.dataset_C_annotated;
    dataset_type = 'C';
elseif isfield(S, 'dataset_A')
    dataset = S.dataset_A;
    dataset_type = 'A';
elseif isfield(S, 'dataset_B')
    dataset = S.dataset_B;
    dataset_type = 'B';
elseif isfield(S, 'dataset_C')
    dataset = S.dataset_C;
    dataset_type = 'C';
elseif isfield(S, 'dataset')
    dataset = S.dataset;
    dataset_type = detect_dataset_type(dataset);
else
    fields = fieldnames(S);
    dataset = S.(fields{1});
    dataset_type = detect_dataset_type(dataset);
    fprintf('Using variable: %s\n', fields{1});
end

fprintf('Detected dataset type: %s\n', dataset_type);

% === Extract data from each trial ===
numTrials = size(dataset, 2);
trials = dataset(2, :);
fprintf('Processing %d trials...\n', numTrials);

trial_data = cell(numTrials, 1);

for i = 1:numTrials
    fprintf('Processing trial %d/%d\n', i, numTrials);
    T = trials{i};
    
    if strcmp(dataset_type, 'A')
        p_ves = T(:, 4);   % P_ves data
        abd = T(:, 15);    % abd annotations
        void = T(:, 16);   % void annotations
        do = T(:, 17);     % do annotations
        
        % Check for invalid annotations and create mask to exclude them
        if size(T, 2) >= 18
            invalid = T(:, 18);
            invalid_mask = invalid == 1;
        else
            invalid_mask = false(size(p_ves));
        end
        
        % Create mask for valid data points (not marked as invalid)
        valid_mask = ~invalid_mask;
        
        % Filter out invalid data points
        p_ves = p_ves(valid_mask);
        abd = abd(valid_mask);
        void = void(valid_mask);
        do = do(valid_mask);
        
        trial_table = table(p_ves, abd, void, do, ...
            'VariableNames', {'P_ves', 'abd', 'void', 'do'});
        
        fprintf('  Trial %d: %d samples (excluded %d invalid), %d abd, %d void, %d do annotations\n', ...
            i, height(trial_table), sum(invalid_mask), sum(abd), sum(void), sum(do));
        
    elseif strcmp(dataset_type, 'B') || strcmp(dataset_type, 'C')
        if istable(T)
            pves = T.Pves;
            abd = T.ABD;
            void = T.VOID;
            do = T.DO;
            
            % Check for invalid annotations
            if any(strcmp(T.Properties.VariableNames, 'INVALID'))
                invalid = T.INVALID;
                invalid_mask = invalid == 1;
            else
                invalid_mask = false(size(pves));
            end
            
        else
            headers = T(1, :);
            data = T(2:end, :);
            
            pves_idx = find(strcmpi(headers, 'Pves'));
            abd_idx = find(strcmpi(headers, 'ABD'));
            void_idx = find(strcmpi(headers, 'VOID'));
            do_idx = find(strcmpi(headers, 'DO'));
            invalid_idx = find(strcmpi(headers, 'INVALID'));
            
            if isempty(pves_idx) || isempty(abd_idx) || isempty(void_idx) || isempty(do_idx)
                error('Could not find required columns (Pves, ABD, VOID, DO) in trial %d', i);
            end
            
            pves = data(:, pves_idx);
            abd = data(:, abd_idx);
            void = data(:, void_idx);
            do = data(:, do_idx);
            
            % Check for invalid annotations
            if ~isempty(invalid_idx)
                invalid = data(:, invalid_idx);
                invalid_mask = invalid == 1;
            else
                invalid_mask = false(size(pves));
            end
        end
        
        % Create mask for valid data points (not marked as invalid)
        valid_mask = ~invalid_mask;
        
        % Filter out invalid data points
        pves = pves(valid_mask);
        abd = abd(valid_mask);
        void = void(valid_mask);
        do = do(valid_mask);
        
        trial_table = table(pves, abd, void, do, ...
            'VariableNames', {'Pves', 'ABD', 'VOID', 'DO'});
        
        fprintf('  Trial %d: %d samples (excluded %d invalid), %d ABD, %d VOID, %d DO annotations\n', ...
            i, height(trial_table), sum(invalid_mask), sum(abd), sum(void), sum(do));
    else
        error('Unknown dataset type: %s', dataset_type);
    end
    
    trial_data{i} = trial_table;
end

% === Save the extracted data ===
default_filename = sprintf('extracted_trial_data_%s.mat', dataset_type);
[save_file, save_path] = uiputfile('*.mat', 'Save extracted data as', default_filename);
if isequal(save_file, 0)
    fprintf('Save cancelled.\n');
    return;
end

save(fullfile(save_path, save_file), 'trial_data', 'numTrials', 'dataset_type');
fprintf('\nExtraction complete!\n');
fprintf('Saved %d trials from dataset %s to: %s\n', numTrials, dataset_type, fullfile(save_path, save_file));

% === Display summary ===
fprintf('\n=== DATASET %s SUMMARY ===\n', dataset_type);
display_summary(trial_data, dataset_type);

% === Optional: Save individual trial files ===
answer = questdlg('Save individual trial files as well?', 'Save Individual Files', 'Yes', 'No', 'No');
if strcmp(answer, 'Yes')
    save_individual_trials(trial_data, save_path, dataset_type);
end

fprintf('\nDone!\n');
end

function dataset_type = detect_dataset_type(dataset)
dataset_type = 'A'; % Default to A
try
    if size(dataset, 2) > 0
        sample_trial = dataset{2, 1};
        if istable(sample_trial)
            var_names = sample_trial.Properties.VariableNames;
            if any(strcmpi(var_names, 'Pves')) && any(strcmpi(var_names, 'ABD'))
                dataset_type = 'B'; % B and C share structure
            end
        else
            if size(sample_trial, 1) > 0
                headers = sample_trial(1, :);
                if any(strcmpi(headers, 'Pves')) && any(strcmpi(headers, 'ABD'))
                    dataset_type = 'B'; % B and C share structure
                end
            end
        end
    end
catch
    fprintf('Warning: Could not auto-detect dataset type. Defaulting to A.\n');
    dataset_type = 'A';
end
end

function display_summary(trial_data, dataset_type)
total_samples = 0;
total_col2 = 0;
total_col3 = 0;
total_col4 = 0;

for i = 1:length(trial_data)
    T = trial_data{i};
    total_samples = total_samples + height(T);
    
    if strcmp(dataset_type, 'A')
        total_col2 = total_col2 + sum(T.abd);
        total_col3 = total_col3 + sum(T.void);
        total_col4 = total_col4 + sum(T.do);
    else
        total_col2 = total_col2 + sum(T.ABD);
        total_col3 = total_col3 + sum(T.VOID);
        total_col4 = total_col4 + sum(T.DO);
    end
end

fprintf('Total samples across all trials: %d\n', total_samples);
if strcmp(dataset_type, 'A')
    fprintf('Total abd annotations: %d\n', total_col2);
    fprintf('Total void annotations: %d\n', total_col3);
    fprintf('Total do annotations: %d\n', total_col4);
else
    fprintf('Total ABD annotations: %d\n', total_col2);
    fprintf('Total VOID annotations: %d\n', total_col3);
    fprintf('Total DO annotations: %d\n', total_col4);
end
end

function save_individual_trials(trial_data, save_path, dataset_type)
numTrials = length(trial_data);
for i = 1:numTrials
    trial_table = trial_data{i};
    filename = sprintf('trial_%s_%03d.mat', dataset_type, i);
    save(fullfile(save_path, filename), 'trial_table');
end
fprintf('Saved %d individual trial files for dataset %s\n', numTrials, dataset_type);
end