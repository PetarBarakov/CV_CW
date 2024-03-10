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

% NOTE: Algorithms that work well on Image (1) and Image (6) are: Harris, 
% MinEigen, ORB (works quite well for features inside of the object)

[features_im1, feature_index_im1 ] = extractFeatures(img1_grey, corners_im1);
[features_im2, feature_index_im2 ] = extractFeatures(img2_grey, corners_im2);

feature_pairs = matchFeatures(features_im1, features_im2);

matched_points_im1 = feature_index_im1(feature_pairs(:, 1), :);
matched_points_im2 = feature_index_im2(feature_pairs(:, 2), :);

figure;
showMatchedFeatures(img1_grey, img2_grey, matched_points_im1, matched_points_im2, "montag")

% Manual Feature Detection
% Note that these Manual points have been computed on the following images:
% source_images\HG_no_grid\WhatsApp Image 2024-03-08 at 18.01.55.jpeg
% source_images\HG_no_grid\WhatsApp Image 2024-03-08 at 18.01.55 (2).jpeg
load manual_points.mat

% [fixedPoints_img1, movingPoints_img2] = cpselect(img1, img2, 'Wait', true) 

showMatchedFeatures(img1, img2, fixedPoints_img1, movingPoints_img2, "montag")

% save manual_points.mat fixedPoints_img1 movingPoints_img2

%% TASK 3: Camera Calibration 
calibration_square_size = 2.2;      % 2.2cm on A4 paper
% cameraCalibrator("source_images\all_1D_grid\", calibration_square_size);
% cameraCalibrator("source_images\1d_grid\", calibration_square_size);


%% TASK 4: Transformation Estimation: Homography
img1_HG = imread("source_images\HG_no_grid\WhatsApp Image 2024-03-08 at 18.01.55 (1).jpeg");
img2_HG = imread("source_images\HG_no_grid\WhatsApp Image 2024-03-08 at 18.01.55 (2).jpeg");

% The code below gives an estimation for the AFFINE transformation and not
% the HOMOGRAPHY matrix. Googling seems that we have to solve the following 
% equation where x' = Hx where x and x' take the form [x, y, 1]. 
% This then gives us a system of equations we can solve to get our
% parameters of H. To solve we need at least 4 matched points. 

% There are also references to using the Direct Linear Transformation
% https://medium.com/@insight-in-plain-sight/estimating-the-homography-matrix-with-the-direct-linear-transform-dlt-ec6bbb82ee2b
% https://www.youtube.com/watch?v=Hi9EFtzPl8Y

% [tform,inlierIdx] = estgeotform2d(matched_points_im1, matched_points_im2,"similarity");
% inlier_pts_im1 = matched_points_im1(inlierIdx,:);
% inlier_pts_im2  = matched_points_im2(inlierIdx,:);
% 
% showMatchedFeatures(img1_grey, img2_grey, inlier_pts_im1, inlier_pts_im2)
% title("Matched Inlier Points")
% 
% outputView = imref2d(size(img1_grey));
% recovered = imwarp(img2_grey,tform,OutputView=outputView);
% 
% figure, imshowpair(img1_grey,recovered,'montage')



% NEVERMIND: I think I was being dumb and we can use the projtform2d()
% function. See link below:
% https://uk.mathworks.com/help/images/matrix-representation-of-geometric-transformations.html

%% TASK 4: Transformation Estimation: Fundamental
img1_FD = imread("source_images\FD_no_grid\WhatsApp Image 2024-03-08 at 18.14.06.jpeg");
img2_FD = imread("source_images\FD_no_grid\WhatsApp Image 2024-03-08 at 18.14.06 (5).jpeg");

img1_FD = im2gray(img1_FD);
img2_FD = im2gray(img2_FD);

% We have to solve the following  equation where x'Fx = 0 where x and x' 
% take the form [x, y, 1]. This then gives us a system of equations we can 
% solve to get our parameters of F. 
% Useful Link: https://uk.mathworks.com/help/vision/ref/epipolarline.html
% Link shows how to calculate fundamental matrix, epipoles and epipolar
% lines 

% Automatic Feature Detection 
% NOTE:
% KAZE seems to detect the most points but all functionsare roughly inaccurate 
corners_im1 = detectKAZEFeatures(img1_FD);
corners_im2 = detectKAZEFeatures(img2_FD);

[features_im1, feature_index_im1 ] = extractFeatures(img1_grey, corners_im1);
[features_im2, feature_index_im2 ] = extractFeatures(img2_grey, corners_im2);

feature_pairs = matchFeatures(features_im1, features_im2);

matched_points_im1 = feature_index_im1(feature_pairs(:, 1), :);
matched_points_im2 = feature_index_im2(feature_pairs(:, 2), :);

figure;
showMatchedFeatures(img1_FD, img2_FD, matched_points_im1, matched_points_im2, "montag")

% Estimate Fundamental Matrix
F = estimateFundamentalMatrix(matched_points_im1, matched_points_im2);

%% TASK 5: 3D Geometry 
% Load Images
img1_FD = imread("source_images\WhatsApp Image 2024-03-10 at 11.38.32 (1).jpeg");
img2_FD = imread("source_images\WhatsApp Image 2024-03-10 at 11.38.32.jpeg");

% Rectifying Images
% 

% Estimating Depth
% NOTE: Can use disprarityBM or disparitySGM

A = stereoAnaglyph(img1_FD,img2_FD);
figure
imshow(A)
title("Red-Cyan composite view of the rectified stereo pair image")

J1 = im2gray(img1_FD);
J2 = im2gray(img2_FD);

disparityRange = [0 48];
disparityMap = disparityBM(J1,J2);

figure
imshow(disparityMap,disparityRange)
title("Disparity Map")
colormap jet
colorbar
