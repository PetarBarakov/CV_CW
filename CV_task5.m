clc
close all

StandardGridImages = {'source_images\1d_grid_pngs\MicrosoftTeams-image.png',...
  'source_images\1d_grid_pngs\MicrosoftTeams-image (1).png', ... 
  'source_images\1d_grid_pngs\MicrosoftTeams-image (2).png', ... 
  'source_images\1d_grid_pngs\MicrosoftTeams-image (3).png', ... 
  'source_images\1d_grid_pngs\MicrosoftTeams-image (4).png', ... 
  'source_images\1d_grid_pngs\MicrosoftTeams-image (5).png', ... 
  % 'source_images\FD_1D_grid\MicrosoftTeams-image (6).png', ...
  % 'source_images\FD_1D_grid\MicrosoftTeams-image (7).png', ...
  };
            
  % StandardGridImages = {'source_images\Task5_camera_calibration\IMG_20240321_075631.jpg',...
  %   'source_images\Task5_camera_calibration\IMG_20240321_075635.jpg', ... 
  %   'source_images\Task5_camera_calibration\IMG_20240321_075642.jpg', ... 
  %   'source_images\Task5_camera_calibration\IMG_20240321_075644.jpg', ... 
  %   'source_images\Task5_camera_calibration\IMG_20240321_075648.jpg', ... 
  %   'source_images\Task5_camera_calibration\IMG_20240321_075651.jpg', ... 
  %   'source_images\Task5_camera_calibration\IMG_20240321_075654.jpg', ... 
  %   'source_images\Task5_camera_calibration\IMG_20240321_075701.jpg', ... 
  %   'source_images\Task5_camera_calibration\IMG_20240321_075704.jpg', ... 
  % };


imgLeft = imread("source_images\additional_FD\11_03_24.png");
imgRight = imread("source_images\additional_FD\11_03_24 (8).png");

% imgLeft =   imread("source_images\Task5_camera_calibration\IMG_20240321_075631.jpg");
% imgRight =  imread("source_images\Task5_camera_calibration\IMG_20240321_075635.jpg");


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

figure
showMatchedFeatures(imgLeft, imgRight, inlierPointsLeft, inlierPointsRight)

[tform1, tform2] = estimateStereoRectification(fMatrix, inlierPointsLeft.Location, ...
    inlierPointsRight.Location, size(imgRight));

[imgLeftRect, imgRightRect] = rectifyStereoImages(imgLeft,imgRight, tform1,tform2);

epiLinesLeft = epipolarLine(fMatrix ,inlierPointsLeft);
pointsLeft = lineToBorderPoints(epiLinesLeft,size(imgLeftRect));

epiLinesRight = epipolarLine(fMatrix ,inlierPointsRight);
pointsRight = lineToBorderPoints(epiLinesLeft,size(imgRightRect));


imgLeftRectGray = im2gray(imgLeftRect);
imgRightRectGray = im2gray(imgRightRect);

disparityRange = [0 40];
disparityMap = disparityBM(imgLeftRectGray, imgRightRectGray);


% Uncomment for different rectification, poot disparity

% CameraGridParameters = CalibrateFromSquare(StandardGridImages);
% poseCamera = estrelpose(fMatrix, CameraGridParameters.Intrinsics, CameraGridParameters.Intrinsics, inlierPointsLeft, inlierPointsRight);
% stereoCameraParameters = stereoParameters(CameraGridParameters, CameraGridParameters, poseCamera);

% [imgLeftRect, imgRightRect, reprojectionMatrix] = rectifyStereoImages(imgLeft,imgRight, stereoCameraParameters);

% epiLinesLeft = epipolarLine(fMatrix ,inlierPointsLeft);
% pointsLeft = lineToBorderPoints(epiLinesLeft,size(imgLeftRect));

% epiLinesRight = epipolarLine(fMatrix ,inlierPointsRight);
% pointsRight = lineToBorderPoints(epiLinesLeft,size(imgRightRect));

% imgLeftRectGray = im2gray(imgLeftRect);
% imgRightRectGray = im2gray(imgRightRect);

% disparityRange = [0 60];
% disparityMap = disparityBM(imgLeftRectGray, imgRightRectGray);









figure
imshowpair(imgLeftRectGray, imgRightRectGray)


figure
imshow(imgLeftRect)
hold on;
line(pointsLeft(:,[1,3])',pointsLeft(:,[2,4])');

figure
imshow(imgRightRect)
hold on;
line(pointsRight(:,[1,3])',pointsRight(:,[2,4])');

figure
imshow(disparityMap, disparityRange)
% title("Disparity Map")
colormap jet
colorbar







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