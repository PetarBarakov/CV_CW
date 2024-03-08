%% ----------------TASK 2: Keypoint corrspondences between images ----------
clear all
close all
clc

img1 = imread("source_images\no_grid\MicrosoftTeams-image (1).png");
img2 = imread("source_images\no_grid\MicrosoftTeams-image (2).png");

img1_grey = transpose(im2gray(img1));
img2_grey = transpose(im2gray(img2));

%------------------ Feature Detection ---------
corners_im1 = detectSURFFeatures(img1_grey);
corners_im2 = detectSURFFeatures(img2_grey);

% NOTE: Algorithms that work well on Image (1) and Image (6) are: Harris, MinEigen,
%       ORB (works quite well for features inside of the object)

[features_im1, feature_index_im1 ] = extractFeatures(img1_grey, corners_im1);
[features_im2, feature_index_im2 ] = extractFeatures(img2_grey, corners_im2);

feature_paris = matchFeatures(features_im1, features_im2);

matched_poitns_im1 = feature_index_im1(feature_paris(:, 1), :);
matched_poitns_im2 = feature_index_im2(feature_paris(:, 2), :);

figure;
showMatchedFeatures(img1_grey, img2_grey, matched_poitns_im1, matched_poitns_im2, "montag")

% %Manual feature point selection
% [fixedPoints_img1, movingPoints_img2] = cpselect(img1, img2, 'Wait', true) 
% showMatchedFeatures(img1, img2, fixedPoints_img1, movingPoints_img2, "montag")

%% ----------------TASK 3: Camera Calibration ---------------------------
calibration_square_size = 2.2;      %2.2cm on A4 paper
% cameraCalibrator("source_images\all_1D_grid\", calibration_square_size);
cameraCalibrator("source_images\FD_1D_grid\", calibration_square_size);


%% ----------------Task 4: Transformation estimation --------------------




%% ----------------Task 5: 3D Geometry ----------------------------------