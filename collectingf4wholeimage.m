% Read the image
image_2 = imread('flower.jpeg');
image_1 =imread('blackandwhite.jpeg');
% imageMatrix = zeros(50, 50);

flower_section = image_2(147:248,334:435);
squares_section = image_1(1:101,1:101);

% % Define the number of rows and columns for the white region
% whiteRows = 50;
% whiteCols = 25;  % Half of the columns
% 
% % Set the specified region to ones (white)
% imageMatrix(:, 1:whiteCols) = 1;


% grayImage2 = rgb2gray(imageMatrix); %this function takes any image of colour 
%or not and converts it to a greyscale image. 

convert_img = rgb2gray(image_2);
distorted_image = imnoise(flower_section, 'salt & pepper'); %distort the image
distorted_image1 = imnoise(squares_section, 'salt & pepper');
% figure
% imshow(distorted_image)
figure
imshow(distorted_image1)
window_size = 5;
pixelMatrix = double(distorted_image1);
% Pad the matrix to handle edge cases
paddedMatrix = padarray(pixelMatrix, [window_size-1, window_size-1], 0, 'both');

% Calculate the variance for all 5x5 windows
windowVariances = nlfilter(paddedMatrix, [window_size window_size], @(x) var(x(:)));

% Find the top-left indices of the 5x5 matrix with the largest variance
[maxVariance, maxIndex] = max(windowVariances(:));

% Convert the linear index to subscripts
[topLeftRowIndex, topLeftColIndex] = ind2sub(size(windowVariances), maxIndex);

% Adjust indices to ensure they are within bounds
topLeftRowIndex = max(1, min(topLeftRowIndex, size(pixelMatrix, 1) - window_size + 1));
topLeftColIndex = max(1, min(topLeftColIndex, size(pixelMatrix, 2) - window_size + 1));
% Extract the 5x5 matrix with the largest variance
maxVarianceMatrix = pixelMatrix(topLeftRowIndex:topLeftRowIndex+window_size-1, ...
    topLeftColIndex:topLeftColIndex+window_size-1);

% Define the 5x5 matrix as the mask
mask = maxVarianceMatrix;

% Get the size of the mask
[maskRows, maskCols] = size(mask);

% Get the size of the image
[imgRows, imgCols] = size(distorted_image);

cell_of_matrices = cell(size(distorted_image));

% Iterate over each position in the mask
for i = 1:imgRows
    for j = 1:imgCols
        % Shift the mask so that the current position becomes the central point (3,3)
        shiftedMask = circshift(mask, [3-i, 3-j]);

        % Calculate the starting and ending indices for the shifted mask
        startRow = 1;
        startCol = 1;
        endRow = startRow + imgRows - 1;
        endCol = startCol + imgCols - 1;

        % Ensure the indices are within bounds
        endRow = min(imgRows, endRow);
        endCol = min(imgCols, endCol);

        % Extract the sub-image corresponding to the current shifted mask position
        subImage = double(distorted_image(startRow:endRow, startCol:endCol));
                % Store the results in a cell array
         cell_of_matrices{i, j} = struct('ShiftedMask', shiftedMask, 'SubImage', subImage);
  
    end
end

f0_values = zeros(imgRows, imgCols);


for i=1:imgRows
    for j=1:imgCols
        f0_values(i,j) = cell_of_matrices{i,j}.ShiftedMask(3,3);
    end
end


gf_matrix = zeros(5,5); %creating empty 5x5 to store the gradient field for each position
cell_of_grads = cell(maskRows, maskCols); %create cell to store every approx_grad
cell_of_smoothed_grads = cell(maskRows, maskCols); % create cell to store smoothed approx_grads
cell_of_M_values = cell(maskRows, maskCols);


% Display the results for each shifted mask position
for i = 1:imgRows
    for j = 1:imgCols
        disp(['Shifted Mask at Position (' num2str(i) ',' num2str(j) '):']);
        disp('Shifted Mask:');
        disp(cell_of_matrices{i,j}.ShiftedMask);
        disp('---------------------');
        approx_grad = differentials2(cell_of_matrices{i,j}.SubImage); % call differential function to compute 
        % the gradient field matrix
        approx_grad_1(i,j) = approx_grad(1);
        approx_grad_2(i,j) = approx_grad(2);
        approx_grad_3(i,j) = approx_grad(3);
        approx_grad_4(i,j) = approx_grad(4);
        approx_grad_5(i,j) = approx_grad(5);
        % M = reconstruct2(smoothed_approx_grad);
        % cell_of_M_values{i,j} = M;
        % gf_matrix(i, j) = approx_grad(1); % extract the first derivative of the grad field
        % cell_of_grads{i, j} = approx_grad(:);
        % smoothed_approx_grad = imgaussfilt(approx_grad, 2); % smoothing image with gaussian filter
        % cell_of_smoothed_grads{i, j} = smoothed_approx_grad; %stores each
        % smoothed grad in a cell
    end
end
smoothed_approx_grad_1 = imgaussfilt(approx_grad_1);
smoothed_approx_grad_2 = imgaussfilt(approx_grad_2);
smoothed_approx_grad_3 = imgaussfilt(approx_grad_3);
smoothed_approx_grad_4 = imgaussfilt(approx_grad_4);
smoothed_approx_grad_5 = imgaussfilt(approx_grad_5);

new_cell_smoothed = cell(imgRows, imgCols);

max_value1=0;
min_value1=0;
for i=1:imgRows
    for j=1:imgCols
        new_cell_smoothed{i,j} = [smoothed_approx_grad_1(i,j); smoothed_approx_grad_2(i,j); smoothed_approx_grad_3(i,j); smoothed_approx_grad_4(i,j); smoothed_approx_grad_5(i,j)];
    end
end

global centre
global nrows
global ncols

all_values = cell(imgRows,imgCols);
[cellRows, cellCols] = size(new_cell_smoothed);

for i=1:cellRows
    for j=1:cellCols
        all_values{i,j} = reconstruct2(new_cell_smoothed{i,j});
    end
end
fn_values = cell(imgRows,imgCols);

for i=1:imgRows
    for j=1:imgCols
        fn_values{i,j} = all_values{i,j}+f0_values(i);
    end
end
reconstructed_matrix = zeros(imgRows, imgCols);

for i = 1:imgRows
    for j = 1:imgCols
        % Use the vector from fn_values directly
        fn_vector = fn_values{i, j};

        fn_vector_with_f0 = [fn_vector(1:12); f0_values(i, j); fn_vector(13:end)];

        % Reshape the vector to match the size of the shifted mask (assuming it's 5x5)
        reshaped_vector = reshape(fn_vector_with_f0, [5, 5]);

        % Shift the reshaped vector to the correct position (3, 3)
        shifted_vector = circshift(reshaped_vector, [3-i, 3-j]);

        % Assign the values to the corresponding positions in the reconstructed matrix
        reconstructed_matrix(i, j) = shifted_vector(3, 3) + f0_values(i, j);
    end
end
max_reconstructed = max(reconstructed_matrix(:));
min_reconstructed = min(reconstructed_matrix(:));
normalized_reconstructed = uint8(255 * (reconstructed_matrix - min_reconstructed) / (max_reconstructed - min_reconstructed));
figure
subplot(1,2,1),imshow(normalized_reconstructed)
subplot(1,2,2),imshow(reconstructed_matrix,[])
% grayscale_pic = im2gray(reconstructed_matrix);
% imshow(grayscale_pic)
% f_0_values = zeros(imgRows, imgCols);
% f_n_values = cell(imgRows, imgCols);
% 
% % Iterate over each point in the matrix
% for i = 1:imgRows
%     for j = 1:imgCols
%         % Get the approximate gradient and M values
%         approx_grad = new_cell_smoothed{i,j};
%         M_values = all_values{i,j};
% 
%         % Calculate the new f_0 value using the approximate gradient
%         new_f_0 = calculate_new_f_0(approx_grad);
% 
%         % Reconstruct f_n values by adding differences to f_0
%         new_f_n_values = new_f_0 + M_values;
% 
%         % Store the f_0 and f_n values
%         f_0_values(i, j) = new_f_0;
%         f_n_values{i,j} = new_f_n_values;
%     end
% end
% 

% f_0_values = zeros(imgRows, imgCols);
% f_n_values = cell(imgRows, imgCols);
% 
% % Iterate over each point in the matrix
% for i = 1:imgRows
%     for j = 1:imgCols
%         % Get the approximate gradient and M values
%         approx_grad = differentials2(cell_of_matrices{i,j}.SubImage);
%         M_values = all_values{i,j};
% 
%         % Calculate the new f_0 value using the approximate gradient
%         new_f_0 = calculate_new_f_0(approx_grad);
% 
%         % Store the f_0 value
%         f_0_values(i, j) = new_f_0;
% 
%         % Store the f_n values
%         f_n_values{i,j} = M_values;
%     end
% end
% 
% f_0_values
% f_n_values

% for i =1:cellRows
%     for j=1:cellCols
%         current=reconstruct2(new_cell_smoothed{i,j});
%         all_values = [all_values; current];
%     end
% end
% min_val = min(all_values);
% max_val = max(all_values);
% 
% scaled_values = 255 * (all_values - min_val) / (max_val - min_val);
% reshaped_matrix = reshape(scaled_values, cellRows, cellCols)



% current= reconstruct2(new_cell_smoothed{centre(1),centre(2)});
% for i = 1:min(imgRows-1, length(current))
%     if current(i) > max_value1
%         max_value1 = current(i);
%     end
%     if current(i) < min_value1
%         min_value1 = current(i);
%     end
% end
% for i=1:((imgRows*imgCols)-1)
%     if current(i)>max_value1
%            max_value1=current(i);
%     end
%         if current(i)<min_value1
%             min_value1=current(i);
%         end
% end
% 
% middleIndex=length(current)/2;
% current = [current(1:middleIndex-1); 0; current(middleIndex:end)];
% 
% scaled_values = 255 * (current - min_value1) / (max_value1 - min_value1);
% 
% if length(scaled_values) == imgRows * imgCols
% 
%     reshaped_matrix = reshape(scaled_values, imgRows, imgCols).';
% 
%     a=(reshaped_matrix)
% 
% else
% 
%     disp('Error: The length of the vector is not compatible with the specified matrix size.');
% 
% end