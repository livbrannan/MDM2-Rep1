the_image = imread('circle.jpeg'); %this function takes the image, 
%and reads it into matlab
image_2 = imread('flower.jpeg');

grayImage = rgb2gray(the_image); %this function takes any image of colour 
%or not and converts it to a greyscale image. 
grayImage2 = rgb2gray(image_2);

smoothedImage = imgaussfilt(grayImage, 2); %this function smooths out the 
% image, removing any high frequency noise - there are a few different
% filters we can try as Alberto said

smoothedImage2 = imgaussfilt(grayImage2, 2);

edges = edge(smoothedImage, 'Canny'); %this function locates the edges of 
% the primary picture within the image
edges2 = edge(smoothedImage2, 'sobel');

figure
subplot(1, 3, 1), imshow(the_image), title('Original Image');
subplot(1, 3, 2), imshow(smoothedImage), title('Smoothed Image');
subplot(1, 3, 3), imshow(edges), title('Edges');
subplot(1, 3, 1), imshow(image_2), title('Original Image 2');
subplot(1, 3, 2), imshow(smoothedImage2), title('Smoothed Image');
subplot(1, 3, 3), imshow(edges2), title('Edges');

pixelMatrix = double(grayImage);


%from here this is what Alberto did last time
[Fx,Fy] = gradient(pixelMatrix);
[Fx,Fy] = gradient(grayImage2);
Fmod= (Fx.*Fx + Fy.*Fy).^0.5;
figure; imshow(Fmod,[]);
