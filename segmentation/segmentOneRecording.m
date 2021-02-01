function segmentOneRecording(full_filename)
 
[filepath,name,ext] = fileparts(full_filename);
[audio, Fs] = audioread(full_filename);
B_matrix = load('B_matrix.mat');
pi_vector = load('pi_vector.mat');
total_obs_distribution = load('total_obs_distribution.mat');
assigned_states = runSpringerSegmentationAlgorithm(audio, Fs, B_matrix.B_matrix, pi_vector.pi_vector, total_obs_distribution.total_obs_distribution, false);
save_path = strcat(filepath, '/', name, '_states.mat');
save_path_2 = strcat(filepath, '/', name, '_audio.mat');
save(save_path, 'assigned_states');
save(save_path_2, 'audio');
 
end