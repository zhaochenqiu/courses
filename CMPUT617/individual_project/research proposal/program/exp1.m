clear all
close all
clc

im_pa = 'D:\dataset\DRIVE\DRIVE\training\images';


im = double(imread([im_pa '\24_training.tif' ]));

pos1 = [134 380]
pos2 = [140 350]


patch1 = im(pos1(1) - 20:pos1(1) + 20, pos1(2) - 20:pos1(2) + 20, :)
patch2 = im(pos2(1) - 20:pos2(1) + 20, pos2(2) - 20:pos2(2) + 20, :)


center1 = im(pos1(1), pos1(2), :);
center2 = im(pos2(1), pos2(2), :);

subpatch1 = patch1;
subpatch2 = patch2;

for i = 1:3
    subpatch1(:, :, i) = patch1(:, :, i) - center1(i)
    subpatch2(:, :, i) = patch2(:, :, i) - center2(i)
end

[row column byte] = size(subpatch1);



data1 = reshape(subpatch1(:, :, 1), 1, row*column);
data2 = reshape(subpatch2(:, :, 1), 1, row*column);

figure

set(gcf, 'Color', [1.0 1.0 1.0])
subplot(1,2,1)
hist(data1, -50:4:50)
ylim([1 1500])
xlim([-50 50])
subplot(1,2,2)
hist(data2, -50:4:50)
ylim([1 1500])
xlim([-50 50])




