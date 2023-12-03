% Read the image
image_2 = imread('flower.jpeg');

% Convert to grayscale if it's a color image
if size(image_2, 3) == 3
    grayImage2 = rgb2gray(image_2);
else
    grayImage2 = image_2;
end

% Define the 5x5 matrix as the mask
mask = [
    193   214   222   231   232
    183   192   206   222   231
    188   184   191   204   217
    189   188   185   188   201
    189   191   191   191   192
];

% Get the size of the mask
[maskRows, maskCols] = size(mask);

% Get the size of the image
[imgRows, imgCols] = size(grayImage2);

% Initialize cell array to store F vectors
allFVectors = cell(maskRows, maskCols);

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
        subImage = double(grayImage2(startRow:endRow, startCol:endCol));

        % Calculate the F vector for the current sub-image
        F = subImage(:) - subImage(3, 3);

        % Store the F vector in the cell array
        allFVectors{i, j} = F;
    end
end

% Display the F vectors for each shifted mask position
for i = 1:maskRows
    for j = 1:maskCols
        disp(['Shifted Mask at Position (' num2str(i) ',' num2str(j) '):']);
        disp('Shifted Mask:');
        disp(circshift(mask, [3-i, 3-j]));
        disp('F Vector:');
        disp(allFVectors{i, j});
        disp('---------------------');
    end
end
