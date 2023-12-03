clear all;
im = im2double(imread('flower.jpeg'));
im = double(im);
[nr,nc]=size(im);
[dx,dy] = gradient(im);
[x y] = meshgrid(1:nc,1:nr);
u = dx;
v = dy;
quiver(x,y,u,v)
