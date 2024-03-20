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


%% --------------- Task 5: Stereo Geometry ------------------

close all

imgLeft = imread("source_images\additional_FD\11_03_24.png");
imgRight = imread("source_images\additional_FD\11_03_24 (8).png");

imgLeftGray = im2gray(imgLeft);
imgRightGray = im2gray(imgRight);

% figure;
% imshowpair(imgLeft, imgRight, 'montag');

% figure 
% imgSstereoAnaglyph(imgLeft, imgRight)
% imshow(stereoAnaglyph(imgLeft, imgRight))
% title("Composite Image (Red - Left Image, Cyan - Right Image)")

pointsLeft = detectSURFFeatures(imgLeftGray,MetricThreshold=1000);
pointsRight = detectSURFFeatures(imgRightGray,MetricThreshold=1000);

% figure 
% subplot(1, 2, 1)
% imshow(imgLeft)
% hold on
% plot(selectStrongest(pointsLeft,20))

% subplot(1, 2, 2)
% imshow(imgRight)
% hold on
% plot(selectStrongest(pointsRight,20))

[featuresLeft, validFeaturesLeft] = extractFeatures(imgLeftGray, pointsLeft);
[featuresRight, validFeaturesRight] = extractFeatures(imgRightGray, pointsRight);

indexPairs = matchFeatures(featuresLeft, featuresRight, Metric="SAD", MatchThreshold=5);

matchedPointsLeft = validFeaturesLeft(indexPairs(:,1),:);
matchedPointsRight = validFeaturesRight(indexPairs(:,2),:);

% figure 
% showMatchedFeatures(imgLeft, imgRight, matchedPointsLeft, matchedPointsRight)



[fMatrix, epipolarInliers, status] = estimateFundamentalMatrix(...
  matchedPointsLeft,matchedPointsRight,Method="RANSAC", ...
  NumTrials=10000,DistanceThreshold=0.1,Confidence=99.99);
  
if status ~= 0 || isEpipoleInImage(fMatrix,size(imgLeft)) ...
  || isEpipoleInImage(fMatrix',size(imgRight))
  error(["Not enough matching points were found or "...
         "the epipoles are inside the images. Inspect "...
         "and improve the quality of detected features ",...
         "and images."]);
end

inlierPointsLeft = matchedPointsLeft(epipolarInliers, :);
inlierPointsRight = matchedPointsRight(epipolarInliers, :);

% figure
% showMatchedFeatures(imgLeft, imgRight, inlierPointsLeft, inlierPointsRight)

[tform1, tform2] = estimateStereoRectification(fMatrix, inlierPointsLeft.Location, ...
    inlierPointsRight.Location, size(imgRight));

[imgLeftRect, imgRightRect] = rectifyStereoImages(imgLeft,imgRight, tform1,tform2);

epiLinesLeft = epipolarLine(fMatrix ,inlierPointsLeft);
pointsLeft = lineToBorderPoints(epiLinesLeft,size(imgLeftRect));

epiLinesRight = epipolarLine(fMatrix ,inlierPointsRight);
pointsRight = lineToBorderPoints(epiLinesLeft,size(imgRightRect));


figure
hold on;
title("Stereo Rectified Images")
subplot(1, 2, 1)
imshow(imgLeftRect)
hold on;
line(pointsLeft(:,[1,3])',pointsLeft(:,[2,4])');
subplot(1, 2, 2)
imshow(imgRightRect)
line(pointsRight(:,[1,3])',pointsRight(:,[2,4])');

imgLeftRectGray = im2gray(imgLeftRect);
imgRightRectGray = im2gray(imgRightRect);

disparityRange = [0 40];
disparityMap = disparityBM(imgLeftRectGray, imgRightRectGray);

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