% Read the image
image_2 = imread('flower.jpeg');

grayImage2 = rgb2gray(image_2); %this function takes any image of colour 
%or not and converts it to a greyscale image. 

distorted_image = imnoise(grayImage2, 'salt & pepper'); %distort the image
figure
imshow(distorted_image)

window_size = 5;

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

cell_of_matrices = cell(maskRows, maskCols);

% Iterate over each position in the mask
for i = 1:maskRows
    for j = 1:maskCols
        % Shift the mask so that the current position becomes the central point (3,3)
        shiftedMask = circshift(mask, [3-i, 3-j]);

        % Calculate the starting and ending indices for the shifted mask
        startRow = 1;
        startCol = 1;
        endRow = startRow + maskRows - 1;
        endCol = startCol + maskCols - 1;

        % Ensure the indices are within bounds
        endRow = min(imgRows, endRow);
        endCol = min(imgCols, endCol);

        % Extract the sub-image corresponding to the current shifted mask position
        subImage = double(distorted_image(startRow:endRow, startCol:endCol));

        % Store the results in a cell array
         cell_of_matrices{i, j} = struct('ShiftedMask', shiftedMask, 'SubImage', subImage);
  
    end
end

gf_matrix = zeros(5,5); %creating empty 5x5 to store the gradient field for each position
cell_of_grads = cell(maskRows, maskCols); %create cell to store every approx_grad
cell_of_smoothed_grads = cell(maskRows, maskCols); % create cell to store smoothed approx_grads

% Display the results for each shifted mask position
for i = 1:maskRows
    for j = 1:maskCols
        disp(['Shifted Mask at Position (' num2str(i) ',' num2str(j) '):']);
        disp('Shifted Mask:');
        disp(cell_of_matrices{i,j}.ShiftedMask);
        disp('---------------------');
        approx_grad = differentials(cell_of_matrices{i,j}.ShiftedMask); % call differential function to compute 
        % the gradient field matrix
        gf_matrix(i, j) = approx_grad(1); % extract the first derivative of the grad field
        cell_of_grads{i, j} = approx_grad(:);
        smoothed_approx_grad = imgaussfilt(approx_grad, 2); % smoothing image with gaussian filter
        cell_of_smoothed_grads{i, j} = smoothed_approx_grad;
    end
end
cell_of_dms = reconstruction(cell_of_smoothed_grads)
