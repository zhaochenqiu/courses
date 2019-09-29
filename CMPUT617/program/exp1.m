clear all
close all
clc

x = -10:0.1:10;
y = x;


for i = 1:max(size(x))
    y(i) = sigmoid(x(i));
end

figure
plot(x, y, '.')



function re_value = sigmoid(x)
    re_value = 1/(exp(-x) + 1);
end



