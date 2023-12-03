image_2 = imread('flower.jpeg');

if size(image_2, 3) == 3

    grayImage2 = rgb2gray(image_2);

else

    grayImage2 = image_2;

end

smoothedImage2 = imgaussfilt(grayImage2, 2);

edges2 = edge(smoothedImage2, 'sobel');

 

pixelMatrix = double(grayImage2);

 

% Assuming pixelMatrix is already defined

 

% Constants

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

 

% Display the result

disp('Top-left indices of the 5x5 matrix with the largest amount of varying pixel intensity:');

disp([topLeftRowIndex, topLeftColIndex]);

 

% Display the 5x5 matrix

disp('5x5 Matrix with the largest amount of varying pixel intensity:');

disp(maxVarianceMatrix);