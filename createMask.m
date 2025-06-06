function [BW,maskedRGBImage] = createMask(RGB)
%  [BW,MASKEDRGBIMAGE] = createMask(RGB) thresholds image RGB, The colorspace and
%  minimum/maximum values for each channel of the colorspace were set in the
%  App and result in a binary mask BW and a composite image maskedRGBImage,
%  which shows the original RGB image values under the mask BW.
% Convert RGB image to chosen color space
I = rgb2hsv(RGB);

% Define thresholds for channel 1 based on histogram settings
channel1Min = 0.056;
channel1Max = 0.136;

% Define thresholds for channel 2 based on histogram settings
channel2Min = 0.128;
channel2Max = 1.000;

% Define thresholds for channel 3 based on histogram settings
channel3Min = 0.087;
channel3Max = 0.913;

% Create mask based on chosen histogram thresholds
sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
    (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
    (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
BW = sliderBW;

% Initialize output masked image based on input image.
maskedRGBImage = RGB;

% Set background pixels where BW is false to zero.
maskedRGBImage(repmat(~BW,[1 1 3])) = 0;

end
