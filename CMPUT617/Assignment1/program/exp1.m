clear all
close all
clc

im = imread('../imgs/in000001_pos.tif');


addpath('D:\projects\matrix\common')
addpath('D:\projects\matrix\common_c')





[row column byte] = size(im)

num = 200;

left = rand(num, 1)*column;
top = rand(num, 1)*row;

left = round(left);
top = round(top);

radius = 60;

right = left + radius;
bottom = top + radius;

pos = [left top right bottom];

color = rand(num, 3);
color = round(color*255);



showim = drawRect_plus(im, pos, 16, color);


figure
subplot(1,2,1)
imshow(uint8(im))
subplot(1,2,2)
imshow(uint8(showim))
