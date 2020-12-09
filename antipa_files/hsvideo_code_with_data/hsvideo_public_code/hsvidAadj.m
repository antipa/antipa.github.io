function x = hsvidAadj(y, Hconj, shutter_indicator, pad2d)
% pad2d
x = pad2d(y);
% back project
x = repmat(x,[1,1,size(shutter_indicator,3)]);
% window
x = x.*shutter_indicator;
% filter by conjugate kernel
x = real(fftconv2d(x, Hconj));
end