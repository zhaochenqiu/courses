clear all
close all
clc


C =     [0.0500,  0.0300,  0.0100,
        0.0400, -0.1100,  0.0200,
        0.0700,  0.0800,  0.0600]


E1 = expm(C)



H = zeros(size(C));
A = eye(size(C));
k = 1;

for i = 1:10
    i
   H = H + A
   A = C*A/k
   k = k+1;
end

E3 = H
E1
