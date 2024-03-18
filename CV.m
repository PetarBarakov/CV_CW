%% --------------------
clear all
close all
clc

%% TASK 2: Keypoint Corrspondences Between Images 
img1 = imread("source_images\HG_no_grid\WhatsApp Image 2024-03-08 at 18.01.55.jpeg");
img2 = imread("source_images\HG_no_grid\WhatsApp Image 2024-03-08 at 18.01.55 (2).jpeg");

img1_grey = im2gray(img1);
img2_grey = im2gray(img2);

% Automatic Feature Detection 
corners_im1 = detectSURFFeatures(img1_grey);
corners_im2 = detectSURFFeatures(img2_grey);

% NOTE: For the image set of source_images\HG_no_grid\WhatsApp Image 2024-03-08 at 18.01.55.jpeg
%       and source_images\HG_no_grid\WhatsApp Image 2024-03-08 at 18.01.55 (2).jpeg
%       the best (and visually accurate) feature detection is SURF


[features_im1, feature_index_im1 ] = extractFeatures(img1_grey, corners_im1);
[features_im2, feature_index_im2 ] = extractFeatures(img2_grey, corners_im2);

feature_pairs = matchFeatures(features_im1, features_im2);

matched_points_im1 = feature_index_im1(feature_pairs(:, 1), :);
matched_points_im2 = feature_index_im2(feature_pairs(:, 2), :);

figure;
showMatchedFeatures(img1_grey, img2_grey, matched_points_im1, matched_points_im2, "montag")

% Manual Feature Detection
% Note that these Manual points have been computed on the following images:
% source_images\HG _no_grid\WhatsApp Image 2024-03-08 at 18.01.55.jpeg
% source_images\HG_no_grid\WhatsApp Image 2024-03-08 at 18.01.55 (2).jpeg
load manual_points.mat

% [fixedPoints_img1, movingPoints_img2] = cpselect(img1, img2, 'Wait', true) 
% save manual_points.mat fixedPoints_img1 movingPoints_img2
    
figure;
showMatchedFeatures(img1_grey, img2_grey, fixedPoints_img1, movingPoints_img2, "montag")

%% TASK 3: MANUAL Camera Calibration 
calibration_square_size = 22;      % 22mm on A4 paper
% load("camera_params.mat");
% cameraCalibrator("source_images\all_1D_grid\", calibration_square_size);
cameraCalibrator("source_images\1d_grid\", calibration_square_size);


%% TASK 3: AUTOMATIC Camera Calibration 
% Define images to process
StandardGridImages = {'source_images\1d_grid\WhatsApp Image 2024-03-08 at 18.23.30 (1).jpeg',...
    'source_images\1d_grid\WhatsApp Image 2024-03-08 at 18.23.30.jpeg',...
    'source_images\1d_grid\WhatsApp Image 2024-03-08 at 18.23.31 (1).jpeg',...
    'source_images\1d_grid\WhatsApp Image 2024-03-08 at 18.23.31.jpeg',...
    'source_images\1d_grid\WhatsApp Image 2024-03-08 at 18.23.32 (1).jpeg',...
    'source_images\1d_grid\WhatsApp Image 2024-03-08 at 18.23.32.jpeg',...
    };

CameraGridParameters = CalibrateFromSquare(StandardGridImages);


% Use same imagaes as in task 2
img1 = imread("source_images\HG_no_grid\WhatsApp Image 2024-03-08 at 18.01.55.jpeg");  
img1 = imresize(img1, [2048, 1536]);
img1_undistorted = undistortImage(img1, CameraGridParameters);

figure
montage({img1, img1_undistorted})
% title("Distortion of the camera")


% cameraCalibrator("source_images\wide_angle\wide_angle_grid\", squareSize);
WideAngleGridImages = {'source_images\wide_angle\wide_angle_grid\IMG_20240316_144831.jpg',...
    'source_images\wide_angle\wide_angle_grid\IMG_20240316_144833.jpg',...
    'source_images\wide_angle\wide_angle_grid\IMG_20240316_144836.jpg',...
    'source_images\wide_angle\wide_angle_grid\IMG_20240316_144838.jpg',...
    'source_images\wide_angle\wide_angle_grid\IMG_20240316_144842.jpg',...
    'source_images\wide_angle\wide_angle_grid\IMG_20240316_144845.jpg',...
    'source_images\wide_angle\wide_angle_grid\IMG_20240316_144847.jpg',...
    'source_images\wide_angle\wide_angle_grid\IMG_20240316_144854.jpg',...
    };

WideAngleGridParameters = CalibrateFromSquare(WideAngleGridImages);
img_wide=imread("source_images\wide_angle\IMG_20240316_145002.jpg");
img_wide_undistorted = undistortImage(img_wide, WideAngleGridParameters);

figure
montage({img_wide, img_wide_undistorted})
% title("Distortion of a Wide-Angle Camera")


%% TASK 4: Transformation Estimation: Homography
ref_HG = imread("source_images\HG_no_grid\WhatsApp Image 2024-03-08 at 18.01.55 (1).jpeg");
warp_HG = imread("source_images\HG_no_grid\WhatsApp Image 2024-03-08 at 18.01.55 (2).jpeg");

ref_grey = im2gray(ref_HG);
warp_grey = im2gray(warp_HG);

% Automatic Feature Detection 
corners_ref = detectSURFFeatures(ref_grey);
corners_warp = detectSURFFeatures(warp_grey);

[features_ref, feature_index_ref] = extractFeatures(ref_grey, corners_ref);
[features_warp, feature_index_warp] = extractFeatures(warp_grey, corners_warp);

feature_pairs = matchFeatures(features_ref, features_warp);

matched_points_ref = feature_index_ref(feature_pairs(:, 1));
matched_points_warp = feature_index_warp(feature_pairs(:, 2));

% Keep All Points and Transform:
figure;
showMatchedFeatures(ref_grey, warp_grey, matched_points_ref, matched_points_warp, "montag")

[tform_HG, inlierIdx] = estgeotform2d(matched_points_warp, matched_points_ref,"projective");

figure;
showMatchedFeatures(ref_grey, warp_grey, matched_points_ref, matched_points_warp)
title("All Matched Features")

outputView = imref2d(size(ref_grey));
recovered = imwarp(warp_grey,tform_HG,OutputView=outputView);

figure;
imshowpair(ref_grey,recovered)
title("Overlay of Ref Image vs Transformed Image")

figure;
imshowpair(warp_grey,recovered)
title("Overlay of Original Image vs Transformed Image")

% Removes Outliers and Transform:
matched_points_ref_inliers = matched_points_ref(inlierIdx,:);
matched_points_warp_inliers = matched_points_warp(inlierIdx,:);

figure;
showMatchedFeatures(ref_grey, warp_grey, matched_points_ref_inliers, matched_points_warp_inliers)
title("Matched Inlier Points")

tform_HG_inliers = fitgeotform2d(matched_points_warp_inliers.Location, matched_points_ref_inliers.Location,"projective");
recovered_inliers = imwarp(warp_grey,tform_HG_inliers,OutputView=outputView);

figure;
imshowpair(ref_grey,recovered_inliers)
title("Overlay of Ref Image vs Transformed Image - INLIERS")

%% TASK 4: Transformation Estimation: Fundamental
% TODO: Epipoles, vanishing points and horizon
% TODO: Find out how many outliers the estimation method tolerates
left_FD = imread("source_images\additional_FD\11_03_24 (1).png");
right_FD = imread("source_images\additional_FD\11_03_24  (2).png");

left_FD = im2gray(left_FD);
right_FD = im2gray(right_FD);

% Automatic Feature Detection 
corners_left = detectSURFFeatures(left_FD);
corners_right = detectSURFFeatures(right_FD);

[features_left, feature_index_left] = extractFeatures(left_FD, corners_left);
[features_right, feature_index_right] = extractFeatures(right_FD, corners_right);

feature_pairs = matchFeatures(features_left, features_right);

matched_points_left = feature_index_left(feature_pairs(:, 1));
matched_points_right = feature_index_right(feature_pairs(:, 2));

% Estimate Fundamental Matrix
% Compute the fundamental matrix. It uses the least median of squares 
% method to find the inliers. There is option to use RANSAC too
[fLMedS,inliers] = estimateFundamentalMatrix(matched_points_left, matched_points_right,'NumTrials',4000);

matched_points_left_inliers = matched_points_left(inliers,:);
matched_points_right_inliers = matched_points_right(inliers,:);

% First Image
% Show the inliers in the first image.
figure; 
subplot(121);
imshow(left_FD); 
title('Matched Points and Epipolar Lines in First Image'); 
hold on;

% Compute the epipolar lines in the first image.
epiLines = epipolarLine(fLMedS',matched_points_right(inliers,:));
points = lineToBorderPoints(epiLines,size(left_FD));
line(points(:,[1,3])',points(:,[2,4])');

% Valid epipole logical, specified as true when the image contains an 
% epipole, and false when the image does not contain an epipole. When the 
% image planes are at a great enough angle to each other, you can expect 
% the epipole to be located in the image. When the image planes are at a 
% more subtle angle to each other, you can expect the epipole to be 
% located outside of the image, (but still in the image plane).
[isIn,epipole] = isEpipoleInImage(fLMedS, size(left_FD));
scatter(epipole(:, 1), epipole(:, 2), 'MarkerFaceColor', 'b')

% Plot Matched Points
scatter(matched_points_left_inliers.Location(:, 1), matched_points_left_inliers.Location(:, 2), 'MarkerFaceColor', 'g')

% Second Image
subplot(122); 
imshow(right_FD);
title('Matched Points and Epipolar Lines in Second Image'); hold on;

% Plot Epipolar Lines
epiLines = epipolarLine(fLMedS,matched_points_left(inliers,:));
points = lineToBorderPoints(epiLines,size(right_FD));
line(points(:,[1,3])',points(:,[2,4])');

% Plot Matched Points 
scatter(matched_points_right_inliers.Location(:, 1), matched_points_right_inliers.Location(:, 2), 'MarkerFaceColor', 'r')


%% TASK 5: 3D Geometry - REV 1

% Rectifying Images
% leftImages = imageDatastore("source_images\left\");
% rightImages = imageDatastore("source_images\right\");

% [imagePoints,boardSize] = detectCheckerboardPoints(leftImages.Files,rightImages.Files);

% squareSizeInMillimeters = 22;  % in units of 'mm'
% worldPoints = generateCheckerboardPoints(boardSize,squareSizeInMillimeters);

% I1 = readimage(leftImages,1);
% I2 = readimage(rightImages,1);
% imageSize = [size(I1,1),size(I1,2)];

% stereoParams = estimateCameraParameters(imagePoints,worldPoints,ImageSize=imageSize);

% [img1_FD_valid, img2_FD_valid] = rectifyStereoImages(img1_FD, img2_FD, stereoParams, OutputView='valid');

% Estimating Depth
% NOTE: Can use disprarityBM or disparitySGM

% A = stereoAnaglyph(img1_FD_valid,img2_FD_valid);

% stereoCameraCalibrator("source_images\left\", "source_images\right\", 22)

% Define images to process
% imageFileNames1 = {'source_images\left\WhatsApp Image 2024-03-08 at 18.23.32 (1).jpeg',...
%     'source_images\left\WhatsApp Image 2024-03-08 at 18.23.32.jpeg',...
%     };
% imageFileNames2 = {'source_images\right\WhatsApp Image 2024-03-08 at 18.23.30.jpeg',...
%     'source_images\right\WhatsApp Image 2024-03-08 at 18.23.31 (1).jpeg',...
%     };

% imageFileNames1 = {'source_images\additional_FD\11_03_24.png'};
% imageFileNames2 = {'source_images\additional_FD\11_03_24 (8).png'};

% % Detect calibration pattern in images
% detector = vision.calibration.stereo.CheckerboardDetector();
% [imagePoints, imagesUsed] = detectPatternPoints(detector, imageFileNames1, imageFileNames2);

% % Generate world coordinates for the planar patten keypoints
% squareSize = 22;  % in units of 'mm'
% worldPoints = generateWorldPoints(detector, 'SquareSize', squareSize);

% % Read one of the images from the first stereo pair
% I1 = imread(imageFileNames1{1});
% [mrows, ncols, ~] = size(I1);

% % Calibrate the camera
% [stereoParams, pairsUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
%     'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
%     'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'mm', ...
%     'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
%     'ImageSize', [mrows, ncols]);



% img1_FD = imread("source_images\additional_FD\11_03_24.png");
% img2_FD = imread("source_images\additional_FD\11_03_24 (8).png");

% A = stereoAnaglyph(img1_FD,img2_FD);
% figure
% imshow(A)
% title("Red-Cyan Composite View of the Rectified Stereo Pair Image")


% img1_FD = imresize(img1_FD, [mrows, ncols]);
% img2_FD = imresize(img2_FD, [mrows, ncols]);
% [J1, J2, reprojectionMatrix] = rectifyStereoImages(img1_FsD, img2_FD, stereoParams);

% % J1 = im2gray(imread("source_images\IMG_20240316_181241.jpg"));
% % J2 = im2gray(imread("source_images\IMG_20240316_181246.jpg"));

% figure
% A = stereoAnaglyph(J1,J2);
% figure
% imshow(A)
% title("Red-Cyan Composite View of the Rectified Stereo Pair Image")

% J1 = im2gray(J1);
% J2 = im2gray(J2);



% disparityRange = [0 100];
% disparityMap = disparityBM(J1,J2);

% figure
% imshow(disparityMap,disparityRange)
% title("Disparity Map")
% colormap jet
% colorbar

%% TASK 5: Stereo Detection

% Auto-generated by stereoCalibrator app on 18-Mar-2024
%-------------------------------------------------------

% leftFileNames = "source_images\IMG_20240318_183610.jpg";
% rightFileNames = "source_images\IMG_20240318_183605.jpg";

% Define images to process
leftFileNames = {'source_images\left_grid_on_wall\WhatsApp Image 2024-03-17 at 14.12.26.jpeg',...
    'source_images\left_grid_on_wall\WhatsApp Image 2024-03-17 at 14.12.28 (1).jpeg',...
    'source_images\left_grid_on_wall\WhatsApp Image 2024-03-17 at 14.12.28.jpeg',...
    'source_images\left_grid_on_wall\WhatsApp Image 2024-03-17 at 14.12.30.jpeg',...
    'source_images\left_grid_on_wall\WhatsApp Image 2024-03-17 at 14.12.31 (1).jpeg',...
    };
rightFileNames = {'source_images\right_grid_on_wall\WhatsApp Image 2024-03-17 at 14.12.27.jpeg',...
    'source_images\right_grid_on_wall\WhatsApp Image 2024-03-17 at 14.12.28 (2).jpeg',...
    'source_images\right_grid_on_wall\WhatsApp Image 2024-03-17 at 14.12.29.jpeg',...
    'source_images\right_grid_on_wall\WhatsApp Image 2024-03-17 at 14.12.31 (2).jpeg',...
    'source_images\right_grid_on_wall\WhatsApp Image 2024-03-17 at 14.12.31.jpeg',...
    };

img_left = imread("source_images\additional_FD\11_03_24.png");
img_right = imread("source_images\additional_FD\11_03_24 (8).png");

% Detect calibration pattern in images
detector = vision.calibration.stereo.CheckerboardDetector();
[imagePoints, imagesUsed] = detectPatternPoints(detector, leftFileNames, rightFileNames);

% Generate world coordinates for the planar patten keypoints
squareSize = 22;  % in units of 'mm'
worldPoints = generateWorldPoints(detector, 'SquareSize', squareSize);

% Read one of the images from the first stereo pair
I1 = imread(leftFileNames{1});
[mrows, ncols, ~] = size(I1);
[mrows, ncols, ~] = size(img_left);

% Calibrate the camera
[stereoParams, pairsUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'mm', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% You can use the calibration data to rectify stereo images.
% I2 = imread(rightFileNames{1});
% figure
% montage({I1, I2})

% [J1, J2] = rectifyStereoImages(I1, I2, stereoParams);

% figure
% montage({J1, J2})


figure
montage({img_left, img_right});

img_left = imresize(img_left, [mrows, ncols]);
img_right = imresize(img_right, [mrows, ncols]);

img_left = im2gray(img_left);
img_right = im2gray(img_right);

[img_left, img_right, ~] = rectifyStereoImages(img_left, img_right, stereoParams);
figure
montage({img_left, img_right});

A = stereoAnaglyph(img_left,img_right);
figure
imshow(A)
title("Red-Cyan Composite View of the Rectified Stereo Pair Image")

disparityRange = [0 100];
disparityMap = disparityBM(img_left, img_right);

figure
imshow(disparityMap,disparityRange)
title("Disparity Map")
colormap jet
colorbar


%% ---------------------- FUNCTIONS ---------------------------------
function cameraParams = CalibrateFromSquare(imageFileNames)

    % Detect calibration pattern in images
    detector = vision.calibration.monocular.CheckerboardDetector();
    [imagePoints, imagesUsed] = detectPatternPoints(detector, imageFileNames);
    imageFileNames = imageFileNames(imagesUsed);

    % Read the first image to obtain image size
    originalImage = imread(imageFileNames{1});
    [mrows, ncols, ~] = size(originalImage);

    % Generate world coordinates for the planar pattern keypoints
    squareSize = 22;  % in units of 'mm'
    worldPoints = generateWorldPoints(detector, 'SquareSize', squareSize);

    % Calibrate the camera
    [cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'mm', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

    % % View reprojection errors
    % h1=figure; showReprojectionErrors(cameraParams);

    % % Visualize pattern locations
    % h2=figure; showExtrinsics(cameraParams, 'CameraCentric');

    % % Display parameter estimation errors
    % displayErrors(estimationErrors, cameraParams);
end