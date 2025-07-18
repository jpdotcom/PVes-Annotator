function SignalAnnotationEditor_Unified()
    % === Dataset Selection ===
    choice = questdlg('Which dataset would you like to annotate?', ...
        'Dataset Selection', 'Dataset A', 'Dataset B', 'Dataset C', 'Dataset A');

    if strcmp(choice, 'Dataset A')
        datasetFile = 'dataset_A_annotated.mat';
        datasetVar = 'dataset_A_annotated';
        isDatasetA = true;
        isDatasetBC = false;
    elseif strcmp(choice, 'Dataset B')
        datasetFile = 'dataset_B_annotated.mat';
        datasetVar = 'dataset_B_annotated';
        isDatasetA = false;
        isDatasetBC = true;
    elseif strcmp(choice, 'Dataset C')
        datasetFile = 'dataset_C_annotated.mat';
        datasetVar = 'dataset_C_annotated';
        isDatasetA = false;
        isDatasetBC = true;
    else
        return; % User cancelled
    end

    % === Load Data ===
    try
        S = load(datasetFile);
        dataset = S.(datasetVar);
        numTrials = size(dataset, 2);
        trials = dataset(2, :);  % Use only second row
        currentTrial = 1;
    catch
        errordlg(['Could not load ' datasetFile '. Please make sure the file exists.']);
        return;
    end

    % === GUI Setup ===
    fig = figure('Name', ['P_ves Event Annotation Editor - ' choice], ...
        'Position', [200, 100, 1200, 700]);

    ax1 = subplot(2, 1, 1, 'Parent', fig);  % Top: P_ves
    ax2 = subplot(2, 1, 2, 'Parent', fig);  % Bottom: P_abd

    uicontrol('Style', 'pushbutton', 'String', 'Previous', ...
        'Position', [30, 20, 80, 40], 'Callback', @(~,~) changeTrial(-1));

    uicontrol('Style', 'pushbutton', 'String', 'Next', ...
        'Position', [120, 20, 80, 40], 'Callback', @(~,~) changeTrial(1));

    uicontrol('Style', 'text', 'String', 'Assign Label:', ...
        'Position', [220, 30, 80, 20]);

    if isDatasetA
        labelOptions = {'abd', 'void', 'do', 'invalid'};
    else
        labelOptions = {'ABD', 'VOID', 'DO', 'INVALID'};
    end

    labelMenu = uicontrol('Style', 'popupmenu', ...
        'String', labelOptions, ...
        'Position', [310, 30, 80, 30]);

    uicontrol('Style', 'pushbutton', 'String', 'Annotate', ...
        'Position', [410, 20, 100, 40], 'Callback', @annotateRegion);

    uicontrol('Style', 'pushbutton', 'String', 'Delete', ...
        'Position', [520, 20, 100, 40], 'Callback', @deleteRegion);

    uicontrol('Style', 'pushbutton', 'String', 'Save All', ...
        'Position', [1000, 20, 100, 40], 'Callback', @saveAll);

    trialLabel = uicontrol('Style', 'text', ...
        'Position', [650, 30, 200, 25], ...
        'FontSize', 12, 'FontWeight', 'bold');

    %% === Start GUI ===
    plotTrial();

    function changeTrial(delta)
        currentTrial = max(1, min(numTrials, currentTrial + delta));
        plotTrial();
    end

    function plotTrial()
        cla(ax1); cla(ax2);
        T = trials{currentTrial};

        if isDatasetA
            pves = T(:, 4);
            pabd = T(:, 2);
            abd_mask = T(:, 15) == 1;
            void_mask = T(:, 16) == 1;
            do_mask = T(:, 17) == 1;
            % Check if invalid column exists, if not create it
            if size(T, 2) < 18
                T(:, 18) = 0;  % Initialize invalid column
                trials{currentTrial} = T;
            end
            invalid_mask = T(:, 18) == 1;
        else
            if istable(T)
                pves = T.Pves;
                pabd = T.Pabd;
                abd_mask = T.ABD == 1;
                void_mask = T.VOID == 1;
                do_mask = T.DO == 1;
                % Check if INVALID column exists, if not create it
                if ~any(strcmp(T.Properties.VariableNames, 'INVALID'))
                    T.INVALID = zeros(height(T), 1);
                    trials{currentTrial} = T;
                end
                invalid_mask = T.INVALID == 1;
            else
                pves = T.Pves;
                pabd = T.Pabd;
                abd_mask = T.ABD == 1;
                void_mask = T.VOID == 1;
                do_mask = T.DO == 1;
                % Check if INVALID field exists, if not create it
                if ~isfield(T, 'INVALID')
                    T.INVALID = zeros(length(T.Pves), 1);
                    trials{currentTrial} = T;
                end
                invalid_mask = T.INVALID == 1;
            end
        end

        t = 1:length(pves);

        axes(ax1);
        plot(ax1, t, pves, 'k'); hold(ax1, 'on');
        ylow1 = min(pves); yhigh1 = max(pves);
        drawRegions(ax1, abd_mask, [0.8 1 0.8], labelOptions{1}, ylow1, yhigh1);
        drawRegions(ax1, void_mask, [1 0.8 0.8], labelOptions{2}, ylow1, yhigh1);
        drawRegions(ax1, do_mask, [0.8 0.8 1], labelOptions{3}, ylow1, yhigh1);
        drawRegions(ax1, invalid_mask, [0.8 0.8 0.8], labelOptions{4}, ylow1, yhigh1);
        title(ax1, sprintf('P_{ves} - Trial %d of %d (%s)', currentTrial, numTrials, choice));
        ylabel(ax1, 'P_{ves}'); hold(ax1, 'off');

        axes(ax2);
        plot(ax2, t, pabd, 'Color', [0.3 0.6 1]);
        title(ax2, 'P_{abd}'); xlabel(ax2, 'Time'); ylabel(ax2, 'P_{abd}');

        trialLabel.String = sprintf('Current Trial: %d', currentTrial);
    end

    function drawRegions(ax, mask, color, label, ylow, yhigh)
        ranges = getRanges(mask);
        for i = 1:size(ranges, 1)
            x1 = ranges(i,1); x2 = ranges(i,2);
            fill(ax, [x1 x2 x2 x1], [ylow ylow yhigh yhigh], ...
                color, 'EdgeColor', 'none', 'FaceAlpha', 0.2);
            text(ax, (x1+x2)/2, yhigh, label, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontWeight', 'bold');
        end
    end

    function annotateRegion(~, ~)
        msgbox('Select a region using the rectangle tool on top plot (P_{ves})');

        axes(ax1);
        rect = drawrectangle(ax1, 'Color', 'm');
        wait(rect);

        pos = round(rect.Position);
        x_start = max(1, pos(1));
        x_end = min(x_start + pos(3), getDataLength());
        selected = labelMenu.String{labelMenu.Value};
        T = trials{currentTrial};

        ylimits = ylim(ax1);
        line(ax1, [x_start x_start], ylimits, 'Color', 'g', 'LineStyle', '--');
        line(ax1, [x_end x_end], ylimits, 'Color', 'g', 'LineStyle', '--');

        if isDatasetA
            labelCols = struct('abd', 15, 'void', 16, 'do', 17, 'invalid', 18);
            col = labelCols.(selected);
            % Ensure the column exists
            if size(T, 2) < col
                T(:, col) = 0;
            end
            T(x_start:x_end, col) = 1;
        else
            % Ensure the field/column exists
            if istable(T)
                if ~any(strcmp(T.Properties.VariableNames, selected))
                    T.(selected) = zeros(height(T), 1);
                end
            else
                if ~isfield(T, selected)
                    T.(selected) = zeros(length(T.Pves), 1);
                end
            end
            T.(selected)(x_start:x_end) = 1;
        end
        trials{currentTrial} = T;
        plotTrial();
    end

    function deleteRegion(~, ~)
        msgbox('Select a region to delete annotation from on top plot (P_{ves})');
        axes(ax1); rect = drawrectangle(ax1, 'Color', 'r'); wait(rect);
        pos = round(rect.Position);
        x_start = max(1, pos(1)); x_end = min(x_start + pos(3), getDataLength());
        selected = labelMenu.String{labelMenu.Value};
        T = trials{currentTrial};

        if isDatasetA
            labelCols = struct('abd', 15, 'void', 16, 'do', 17, 'invalid', 18);
            col = labelCols.(selected);
            % Ensure the column exists
            if size(T, 2) >= col
                T(x_start:x_end, col) = 0;
            end
        else
            % Only delete if the field/column exists
            if istable(T)
                if any(strcmp(T.Properties.VariableNames, selected))
                    T.(selected)(x_start:x_end) = 0;
                end
            else
                if isfield(T, selected)
                    T.(selected)(x_start:x_end) = 0;
                end
            end
        end
        trials{currentTrial} = T;
        plotTrial();
    end

    function len = getDataLength()
        T = trials{currentTrial};
        if isDatasetA
            len = size(T, 1);
        else
            if istable(T)
                len = height(T);
            else
                len = length(T.Pves);
            end
        end
    end

    function saveAll(~, ~)
        for i = 1:numTrials
            dataset{2, i} = trials{i};
        end
        if isDatasetA
            defaultName = 'edited_dataset_A.mat';
        elseif strcmp(choice, 'Dataset B')
            defaultName = 'edited_dataset_B.mat';
        else
            defaultName = 'edited_dataset_C.mat';
        end

        [file, path] = uiputfile(defaultName, 'Save Edited Dataset');
        if isequal(file, 0), return; end

        if isDatasetA
            dataset_A_annotated = dataset;
            save(fullfile(path, file), 'dataset_A_annotated');
        elseif strcmp(choice, 'Dataset B')
            dataset_B_annotated = dataset;
            save(fullfile(path, file), 'dataset_B_annotated');
        else
            dataset_C_annotated = dataset;
            save(fullfile(path, file), 'dataset_C_annotated');
        end
        msgbox('Annotations saved successfully!');
    end

    function ranges = getRanges(mask)
        d = diff([0; mask(:); 0]);
        starts = find(d == 1);
        ends = find(d == -1) - 1;
        ranges = [starts, ends];
    end
end