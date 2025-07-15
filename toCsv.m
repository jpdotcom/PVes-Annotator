function ok = toCsv(filePath)
    % Ensure filePath is a string scalar, then convert to char for load()
    matFile = char(filePath + ".mat");
    data = load(matFile);  % Load the .mat file
    
    % Extract table variable (assuming only one table in file)
    fn = fieldnames(data);
    tableData = data.(fn{1});
    tableData{1,:}
    % Write the table to CSV
   
    writetable(tableData, "/Users/jaypatel/Desktop/APTSUMMER/PVES-classifier/Classification/" + filePath + ".csv");

    ok = true;
end
