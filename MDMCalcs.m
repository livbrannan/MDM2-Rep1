%Define matrices
F = [2;23;31;40;41;-8;1;15;31;40;-3;-7;13;26;-2;-3;-6;-3;10;-2;0;0;0;1];
A = [-2 2; -1 1/2; 0 0; 1 1/2; 2 2; -2 2; -1 1/2; 0 0;1 1/2; 2 2; -2 2; -1 1/2; 1 1/2; 2 2; -2 2; -1 1/2; 0 0; 1 1/2; 2 2; -2 2; -1 1/2; 0 0; 1 1/2; 2 2];

%Step 1 - multiply both sides by A transpose
F1 = A'*F;
A2 = A'*A;

%Step 2 - find the inverse of A transpose times A
inverse_A2 = inv(A2);

%Step 3 - calculate the image gradient at position 0
approx_grad = inverse_A2*F1;