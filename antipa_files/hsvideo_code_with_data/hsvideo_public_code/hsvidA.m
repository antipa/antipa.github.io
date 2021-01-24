function y = hsvidA(x, H, shutter_indicator, crop2d)
% Takes in 3d video, x, and computes the cropped sum of 2d convolutions
% using the spectrum of the psf, H. crop is a handle that computes the
% cropping

% Step 1: filter
x = real(fftconv2d(x,H));
% Step 2: crop (multiply)
x = x.*shutter_indicator;
% Step 3: sum
y = crop2d(sum(x,3));
    