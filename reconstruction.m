function cell_of_dms = reconstruction(approx_grads)
    A = [-2 2; -1 0.5; 0 0; 1 0.5; 2 2; -2 2; -1 0.5; 0 0; 1 0.5; 2 2; -2 2; -1 0.5; 1 0.5; 2 2; -2 2; -1 0.5; 0 0; 1 0.5; 2 2; -2 2; -1 0.5; 0 0; 1 0.5; 2 2];
    C = inv(A' * A) * A';

    [rows, cols] = size(approx_grads);
    cell_of_dms = cell(rows, cols);

    for i = 1:rows
        for j = 1:cols
            dm = approx_grads{i, j}; 
            M = C .* dm;
            cell_of_dms{i, j} = M;
        end
    end
end