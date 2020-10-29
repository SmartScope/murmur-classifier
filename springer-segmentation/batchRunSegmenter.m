function batchRunSegmenter(folder)

folder_ext = strcat(folder, '/*.wav');
fileList = dir(folder_ext);
for i = 1:length(fileList)
    filename = fileList(i).name;
    [filepath,name,ext] = fileparts(filename);
    full_filename = strcat(folder, '/', filename);
    [audio, Fs] = audioread(full_filename);
    B_matrix = load("B_matrix.mat");
    pi_vector = load("pi_vector.mat");
    total_obs_distribution = load("total_obs_distribution.mat");
    assigned_states = runSpringerSegmentationAlgorithm(audio, Fs, B_matrix.B_matrix, pi_vector.pi_vector, total_obs_distribution.total_obs_distribution, false);
    save_path = strcat(folder, '/', name, '_states.mat');
    save_path_2 = strcat(folder, '/', name, '_audio.mat');
    save(save_path, 'assigned_states');
    save(save_path_2, 'audio');
end

end

