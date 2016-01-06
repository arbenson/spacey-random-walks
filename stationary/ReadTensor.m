function [R] = ReadTensor(file_name)

data = dlmread(file_name);
data(:, 1:3) = data(:, 1:3) + 1;  % 0 -> 1 index

N = max(max(data(:, 1:3)));
R = zeros(N, N * N);
for ind = 1:size(data, 1)
    i = data(ind, 1);
    j = data(ind, 2);
    k = data(ind, 3);
    val = data(ind, 4);
    R(i, j + (k - 1) * N) = val;
end
    