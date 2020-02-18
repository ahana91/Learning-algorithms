function [Y,Yder]=relu(V)

Nv = size(V, 1);
Y = max(0, V);
Yder = zeros(Nv, 1);
Yder(V > 0) = 1;