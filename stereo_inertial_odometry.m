%% Stereo-Visual Inertial Odometry
% Sam Weinberg
% 12/31/2019

% This code uses an EKF to estimate the 3D pose of an automous vehicle. It
% can be tested using raw data (synced + rectified grayscale, IMU/GPS) from
% the KITTI Benchmark Suite. The motion model consist of IMU measurements 
% and the observation model consists of sychronized grayscale stereo image
% pairs. The EKF uses SO3/SE3 groups to associate uncertainties with the 
% 3D pose. Refer to Barfoot et al. for further details. 

% Clear and close everything
clear all; close all; dbstop error; clc;
disp('======= Stereo Visual Odometry =======');

% Sequence base directory (change as required)
base_dir = 'C:\Users\Sam\Documents\UTIAS\2nd Year\1st Semester\AER 1513\Project';

% Drive number (change as required)
drive_num = '0036';

%% Motion model parameters

% Load oxts data
oxts = loadOxtsliteData(strcat(base_dir, '\2011_09_26\2011_09_26_drive_', drive_num, '_sync'));

% Load timestamps
timestamps = loadTimestamps(strcat(base_dir, '\2011_09_26\2011_09_26_drive_', drive_num, '_sync\oxts'));
format long
timestamps_num = datevec(timestamps);
timestamps_num = datenum([zeros(length(timestamps),1),zeros(length(timestamps),1),zeros(length(timestamps),1),timestamps_num(:,4), timestamps_num(:,5), timestamps_num(:,6)]);
timestamps_num = (timestamps_num - timestamps_num(1))*10^5; 

% Transform to poses
pose = convertOxtsToPose(oxts);   

% Save IMU data
delta_t = zeros(1, length(oxts) - 1);
vel_x = zeros(1, length(oxts));
vel_y = zeros(1, length(oxts));
vel_z = zeros(1, length(oxts));
accel_x = zeros(1, length(oxts));
accel_y = zeros(1, length(oxts));
accel_z = zeros(1, length(oxts));
omega_x = zeros(1, length(oxts));
omega_y = zeros(1, length(oxts));
omega_z = zeros(1, length(oxts)); 
inputs = zeros(6, 1, length(oxts)); 

% Set up 6x1 inputs array
for i = 1:1:length(oxts)
    if i ~= 1
        delta_t(i - 1) = timestamps_num(i) - timestamps_num(i - 1);
        delta_omega(i-1) = omega_z(i) - omega_z(i - 1);
    end
        
    roll(1,i) = oxts{1,i}(4);
    pitch(1,i) = oxts{1,i}(5);
    yaw(1,i) = oxts{1,i}(6);
    vel_x(1,i) = oxts{1,i}(9);
    vel_y(1,i) = oxts{1,i}(10);
    vel_z(1,i) = oxts{1,i}(11);
    accel_x(1,i) = oxts{1,i}(12);
    accel_y(1,i) = oxts{1,i}(13);
    accel_z(1,i) = oxts{1,i}(14);
    omega_x(1,i) = oxts{1,i}(18);
    omega_y(1,i) = oxts{1,i}(19);
    omega_z(1,i) = oxts{1,i}(20); 
    
    inputs(:,1,i) = -[vel_x(1,i); vel_y(1,i); vel_z(1,i); 
                   omega_x(1,i); omega_y(1,i); omega_z(1,i)]; 
end

% cumsum(delta_t*delta_omega.')
% delta_yaw = yaw(1) - yaw(length(yaw))

%% Observation model load parameters

% Load timestamps
timestamps_left = loadTimestamps(strcat(base_dir, '\2011_09_26\2011_09_26_drive_', drive_num, '_sync\image_00'));
timestamps_right = loadTimestamps(strcat(base_dir, '\2011_09_26\2011_09_26_drive_', drive_num, '_sync\image_01'));
timestamps_left_num = datevec(timestamps_left);
timestamps_right_num = datevec(timestamps_right);
timestamps_left_num = datenum([zeros(length(timestamps_left),1),zeros(length(timestamps_left),1),zeros(length(timestamps_left),1),timestamps_left_num(:,4), timestamps_left_num(:,5), timestamps_left_num(:,6)]);
timestamps_right_num = datenum([zeros(length(timestamps_right),1),zeros(length(timestamps_right),1),zeros(length(timestamps_right),1),timestamps_right_num(:,4), timestamps_right_num(:,5), timestamps_right_num(:,6)]);
timestamps_left_num = (timestamps_left_num - timestamps_left_num(1)); 
timestamps_right_num = (timestamps_right_num - timestamps_right_num(1)); 

% Load inverted matrices from KITTI + COnstruct stereo camera model
[veloTOCam, K] = loadCalibration(strcat(base_dir, '\2011_09_26'));
imuTOVelo = loadCalibrationRigid(strcat(base_dir, '\2011_09_26\calib_imu_to_velo.txt'));
calib = loadCalibrationCamToCam(strcat(base_dir, '\2011_09_26\calib_cam_to_cam.txt'));

P_rect = calib.P_rect{1,2}; % Can extract baseline from this one

% Midpoint camera model
M_mid = [P_rect(1:2,1:3) [-P_rect(1,4)/2;0]; 
        P_rect(1:2,1:3) [P_rect(1,4)/2;0]];
    
%veloTOCam = [veloTOCam{1,1}(1:4, 1:3) [(veloTOCam{1,1}(1,4) + veloTOCam{1,2}(1,4))/2; veloTOCam{1,1}(1:3,4)]];
    
% Left camera model
M_left = [P_rect(1:3,1:3) [0;0;P_rect(1,4)]; 
        P_rect(2,1:4)];
    
veloTOCam = veloTOCam{1,1};
 
K_left = calib.K{1,1}; % Camera intrinsic K matrices
K_right = calib.K{1,2};

S_left = calib.S{1,1}; % Covariance matrices
S_right = calib.S{1,2};

camTOVelo = inv(veloTOCam);
veloTOImu = inv(imuTOVelo);

D = [1 0 0 0; 0 1 0 0; 0 0 1 0].';

K_tri = P_rect(1:3,1:3);


%% EKF Initialization

% Process noise
v_var = 1; % [m^2/s^2]
om_x_var = 0.09; % [rad^2/s^2]
om_y_var = 0.13; % [rad^2/s^2]
om_z_var = 0.42; % [rad^2/s^2]

%% EKF Main Loop

tic;
for k = 1:1:length(oxts)
    
    % Load Image Pair
    if k-1 < 10
        Ic_l = imread(strcat(base_dir, '\2011_09_26\2011_09_26_drive_', drive_num, '_sync\image_00\data\000000000', num2str(k-1), '.png'));
        Ic_r = imread(strcat(base_dir, '\2011_09_26\2011_09_26_drive_', drive_num, '_sync\image_01\data\000000000', num2str(k-1), '.png'));
    elseif k-1 < 100
        Ic_l = imread(strcat(base_dir, '\2011_09_26\2011_09_26_drive_', drive_num, '_sync\image_00\data\00000000', num2str(k-1), '.png'));
        Ic_r = imread(strcat(base_dir, '\2011_09_26\2011_09_26_drive_', drive_num, '_sync\image_01\data\00000000', num2str(k-1), '.png'));
    elseif k-1 < 1000
        Ic_l = imread(strcat(base_dir, '\2011_09_26\2011_09_26_drive_', drive_num, '_sync\image_00\data\0000000', num2str(k-1), '.png'));
        Ic_r = imread(strcat(base_dir, '\2011_09_26\2011_09_26_drive_', drive_num, '_sync\image_01\data\0000000', num2str(k-1), '.png'));
    end  
    
    %=========================================================================
    % Step1: Use SURF to find the features on Left camera at t-1 and t 
    
    % previous time step
    % current data is assigned to previous timestep at end of loop

    % current time step
    Pts_cl = detectSURFFeatures(Ic_l);
    [features_cl,valid_Pts_cl] = extractFeatures(Ic_l,Pts_cl);

    %=========================================================================
    % Step2: Use SURF to find the features on Right camera at t-1 and t
    % previous time step
    % current data is assigned to previous timestep at end of loop

    % current time step 
    Pts_cr = detectSURFFeatures(Ic_r);
    [features_cr,valid_Pts_cr] = extractFeatures(Ic_r,Pts_cr);
    
    G = [];
    R = [];
    observations = []; 
    y_check = [];
    
    % Intialize at first timestep
    if k == 1
        % Initialization state and covariance
        T_hat = zeros(4, 4, length(oxts));
        T_hat(:,:,1) = inv(pose{1,1});
        T_hat_cell{1,1} = inv(pose{1,1});
        T_check_cell{1,1} = inv(pose{1,1});
        P_hat = zeros(6, 6, length(oxts));
        P_hat(:,:,1) = diag([1, 1, 1, 0.1, 0.1, 0.1]);    
        Tr_0_inv = inv(T_hat(:,:,1));
        G = [];
        R = [];
        cnt = 0;
        
        T_deadreck{1} = T_hat_cell{1,1};
        
        % Obtain first stereo pair to get landmarks for second timestep
        
        
    else
        %%%%%%%%%%%%%%%%%%%% Motion model %%%%%%%%%%%%%%%%%%%%%%%%
        % Propogate the mean
        
        theta = vec2tran(delta_t(k-1)*inputs(:,:,k));
        T_check = theta*T_hat(:,:,k-1);
        T_deadreck{k} = theta*T_deadreck{1,k-1};

        % Propogate the covariance
        Q = diag([(v_var*delta_t(k-1))^2, (v_var*delta_t(k-1))^2, (v_var*delta_t(k-1))^2, (om_x_var*delta_t(k-1))^2, (om_y_var*delta_t(k-1))^2, (om_z_var*delta_t(k-1))^2]);
        F = tranAd(theta);
        P_check = F*P_hat(:,:,k-1)*F.' + Q;
        
        %%%%%%%%%%%%%%%%%% Observation Model %%%%%%%%%%%%%%%%%%%%%

        %=========================================================================
        % Step 3: use Matchfeature function to pair the previous left and current
        % left image 
        IndexPair_pc = matchFeatures(features_pl,features_cl);
        Matched_Pts_pl = valid_Pts_pl(IndexPair_pc(:,1),:);
        Matched_Pts_cl = valid_Pts_cl(IndexPair_pc(:,2),:);

        Matched_features_pl = features_pl(IndexPair_pc(:,1),:);
        Matched_features_cl = features_cl(IndexPair_pc(:,2),:);

        % RANSAC 
        
       [fRANSAC_cur, inliers_cur] = estimateFundamentalMatrix(Matched_Pts_pl,...
                                  Matched_Pts_cl,'Method','RANSAC',...
                                 'NumTrials',2000,'DistanceThreshold',0.5);
    
        Inlier_pl_pts = Matched_Pts_pl(inliers_cur,:);
        Inlier_cl_pts = Matched_Pts_cl(inliers_cur,:);

        Inlier_features_pl = Matched_features_pl(inliers_cur,:);
        Inlier_features_cl = Matched_features_cl(inliers_cur,:);

        %=========================================================================
        % Step 4: use Matchfeature function to pair the left and right camera
        % features for t and t-1 
        % Previous time step 
        IndexPair_p_lr = matchFeatures(Inlier_features_pl,features_pr);
        % Current time step 
        IndexPair_c_lr = matchFeatures(Inlier_features_cl,features_cr);

        Index_p = IndexPair_p_lr(:,1);
        Index_c = IndexPair_c_lr(:,1);

        [Ip,Ipc] = ismember(Index_p,Index_c);

        Matched_Index_p = IndexPair_p_lr(Ip,1);
        Matched_Index_c = zeros(size(IndexPair_c_lr(:,1)));

        for i = 1:size(Ip)
            if Ipc(i) ~= 0
                Matched_Index_c(Ipc(i)) = 1;
            end
            Ic = logical(Matched_Index_c);
        end

        % We need to compare the left array of IndexPair_p_lr and IndexPair_c_lr
        % so Matched features_pl and Matched feature_cl is paired one to one and
        % we check Matched features_pl(indexPair_p_lr) whether pair Matched
        % features_cl(indexPair_c_lr)
        
        % previous
        Pts_pc_pl = Inlier_pl_pts(IndexPair_p_lr(Ip,1),:);
        Pts_pc_pr = valid_Pts_pr(IndexPair_p_lr(Ip,2),:);
        
        % current
        Pts_pc_cl = Inlier_cl_pts (IndexPair_c_lr(Ic,1),:);
        Pts_pc_cr = valid_Pts_cr(IndexPair_c_lr(Ic,2),:);

        %=========================================================================
        % Step 5:
        % Another RANSAC for left and right image at previous time 
        % RANSAC 
        [fRANSACp__lr, inliersp_lr] = estimateFundamentalMatrix(Pts_pc_pl,...
                                          Pts_pc_pr,'Method','RANSAC',...
                                         'NumTrials',2000,'DistanceThreshold',0.3);
        Pts_pl = Pts_pc_pl(inliersp_lr,:);
        Pts_pr = Pts_pc_pr(inliersp_lr,:);
        
        %figure(1); showMatchedFeatures(Ip_l,Ip_r,Pts_pl,Pts_pr);

        %=========================================================================
        % Step 6:
        % Another RANSAC for left and right image at current time 
        % RANSAC 
        [fRANSACc__lr, inliersc_lr] = estimateFundamentalMatrix(Pts_pc_cl,...
                                          Pts_pc_cr,'Method','RANSAC',...
                                         'NumTrials',2000,'DistanceThreshold',0.3);
        Pts_cl = Pts_pc_cl(inliersc_lr,:);
        Pts_cr = Pts_pc_cr(inliersc_lr,:);
        %figure(2); showMatchedFeatures(Ic_l,Ic_r,Pts_cl,Pts_cr);
        

        Inlier_index = zeros(size(Pts_pc_cl));
        for i = 1:size(Pts_pc_cl)
            if inliersp_lr(i) == inliersc_lr(i) && inliersp_lr(i) == 1 
                Inlier_index(i) = 1;
            end
            Ifinal = logical(Inlier_index);
        end

        Pts_pl_final = Pts_pc_pl(Ifinal,:);
        Pts_pr_final = Pts_pc_pr(Ifinal,:);
        Pts_cl_final = Pts_pc_cl(Ifinal,:);
        Pts_cr_final = Pts_pc_cr(Ifinal,:);
        
        %figure(1); showMatchedFeatures(Ic_l,Ip_l,Pts_cl_final,Pts_pl_final);
        
        % Assign matched points to arrays of pixels for landmark and
        % observation
        landmark_pixels_left = Pts_pl_final.Location;
        landmark_pixels_right = Pts_pr_final.Location;
        
        observation_pixels_left = Pts_cl_final.Location;
        observation_pixels_right = Pts_cr_final.Location;
        
        % Check if there are no common features between timesteps
        if isempty(landmark_pixels_left) == 0
            landmark_pixels_stacked = {};
            landmark_location_stacked = {};

            
            % Run loop for number of observaitons
            cnt = 0;
            for j = 1:1:length(landmark_pixels_left(:,1))
                
                if abs(landmark_pixels_left(j,1) - landmark_pixels_right(j,1)) < 20
                    continue
                end
                
                cnt = cnt + 1;
                
                landmark_pixels_stacked{cnt,1} = [landmark_pixels_left(j,1); 
                                                landmark_pixels_left(j,2); 
                                                landmark_pixels_right(j,1); 
                                                landmark_pixels_right(j,2);];
                                            
                % Convert pixels to world frame
                r_j = (veloTOCam*imuTOVelo*T_hat(:,:,k-1))\triangulate_rect(landmark_pixels_stacked{cnt,1}, K_tri);

                landmark_location_stacked{cnt,1} = [r_j(1:3,1); 1];  

                % Append to G matrix for each observation
                G(3*cnt - 2:3*cnt, 1:6) = D.'*point2fs(T_check*landmark_location_stacked{cnt,1});

                % Measurement model (predicted observations)
                y_check(3*cnt - 2:3*cnt,1) = D.'*(T_check*landmark_location_stacked{cnt,1});

                % Append observations

                observation_pixels_stacked{cnt,1} = [observation_pixels_left(j,1); 
                                                    observation_pixels_left(j,2); 
                                                    observation_pixels_right(j,1); 
                                                    observation_pixels_right(j,2);]; 

                % Convert pixels to vehicle frame 
                y_vj = (veloTOCam*imuTOVelo)\triangulate_rect(observation_pixels_stacked{cnt,1}, K_tri);

                observations(3*cnt - 2:3*cnt,1) = y_vj(1:3,1);
                
                % Append observation noise matrix
                R(3*cnt - 2:3*cnt, 3*cnt - 2:3*cnt) = diag([20, 5.8, 8]);
            end                      
        end
    end
    
            
    % Clear cell arrays for next timestep
    clear landmark_location_stacked landmark_pixels_stacked     
    clear landmark_pixels_left landmark_pixels_right
    clear observation_pixels_left observation_pixels_right
    
    % Assign current features and matches to previous
    Ip_l = Ic_l;
    Ip_r = Ic_r;
    features_pl = features_cl;
    features_pr = features_cr;
    valid_Pts_pl = valid_Pts_cl;
    valid_Pts_pr = valid_Pts_cr;
    
    if cnt == 0 && mod(k,8) == 0
        dummy = 0;
    end
    
    if isempty(R) && k ~= 1
        P_hat(:,:,k) = P_check;        
        T_hat(:,:,k) = (T_check);
        T_hat_cell{1,k} = (T_hat(:,:,k));
    elseif k ~= 1      
        % Kalman Gain
        K = P_check*G.'/(G*P_check*G.' + R);
        
        % Innovation
        innov = observations - y_check;
        
        % Corrector
        P_hat(:,:,k) = (eye(6) - K*G)*P_check; 
        T_hat(:,:,k) = vec2tran(K*innov)*(T_check);
        T_hat_cell{1,k} = (T_hat(:,:,k));
        T_check_cell{1,k} = T_check;
    end  
end

toc;

%% Plotting
% Plot every 10'th pose
for i2=1:1:length(pose)

%   plot3(B2(1,3:4),B2(2,3:4),B2(3,3:4),'-g','LineWidth',2); % y: green
%   plot3(B2(1,5:6),B2(2,5:6),B2(3,5:6),'-b','LineWidth',2); % z: blue
  
  inv_T_deadreck{i2} = inv(T_deadreck{i2});  
  inv_T_hat_cell{i2} = inv(T_hat_cell{i2});
  
  anglez_est = rot2vec(inv_T_deadreck{i2}(1:3,1:3));
  anglez_ground = rot2vec(pose{1,i2}(1:3,1:3));

  % x data points
  x_deadreck(i2) = inv_T_deadreck{i2}(1,4);
  x_est(i2) = inv_T_hat_cell{i2}(1,4);
  x_ground(i2) = pose{1,i2}(1,4);
  x_error(i2) = x_est(i2) - x_ground(i2);
  
  % y data points
  y_deadreck(i2) = inv_T_deadreck{i2}(2,4);
  y_est(i2) = inv_T_hat_cell{i2}(2,4);
  y_ground(i2) = pose{1,i2}(2,4);
  y_error(i2) = y_est(i2) - y_ground(i2);
  
  % z data points
  z_deadreck(i2) = inv_T_deadreck{i2}(3,4);
  z_est(i2) = inv_T_hat_cell{i2}(3,4);
  z_ground(i2) = pose{1,i2}(3,4);
  z_error(i2) = z_est(i2) - z_ground(i2);
   
  % roll data points
  roll_est(i2) = wrapToPi(anglez_est(1));
  roll_ground(i2) = wrapToPi(anglez_ground(1));
  roll_error(i2) = roll_est(i2) - roll_ground(i2);
    
  % pitch data points
  pitch_est(i2) = wrapToPi(anglez_est(2));
  pitch_ground(i2) = wrapToPi(anglez_ground(2));
  pitch_error(i2) = pitch_est(i2) - pitch_ground(i2);
   
  % yaw data points
  yaw_est(i2) = wrapToPi(anglez_est(3));
  yaw_ground(i2) = wrapToPi(anglez_ground(3));
  yaw_error(i2) = wrapToPi(yaw_est(i2) - yaw_ground(i2));  
  
  % variances  
  x_var(i2) = P_hat(1,1,i2);
  y_var(i2) = P_hat(2,2,i2);
  z_var(i2) = P_hat(3,3,i2);
  roll_var(i2) = P_hat(4,4,i2);
  pitch_var(i2) = P_hat(5,5,i2);
  yaw_var(i2) = P_hat(6,6,i2);
end

figure(5); hold on; axis equal;
plot3(x_deadreck, y_deadreck, z_deadreck, '-g') % dead reckon: green
hold on
plot3(x_est, y_est, z_est, '-b'); % estimate: blue
hold on
plot3(x_ground, y_ground, z_ground, 'r'); % ground: red
xlabel('x (m)');
ylabel('y (m)');
zlabel('z (m)');
title('X-Y View KITTI Drive 0036');
legend('Dead Reckoning', 'SVIO', 'Groundtruth');
hold off

% Error plot
figure(6)
plot(timestamps_num, x_error)
hold on
plot(timestamps_num, 3*sqrt(x_var), '--')
hold on
plot(timestamps_num, -3*sqrt(x_var), '--')
legend('x error','x 3\sigma','x -3\sigma')
xlabel('Time [s]')
ylabel('Error [m]')
hold off
%comet(pose{1,:}(1,4),pose{1,:}(2,4))
    
% Error plot
figure(7)
plot(timestamps_num, y_error)
hold on
plot(timestamps_num, 3*sqrt(y_var), '--')
hold on
plot(timestamps_num, -3*sqrt(y_var), '--')
legend('y error','y 3\sigma','y -3\sigma')
xlabel('Time [s]')
ylabel('Error [m]')
hold off

% Error plot
figure(8)
plot(timestamps_num, z_error)
hold on
plot(timestamps_num, 3*sqrt(z_var), '--')
hold on
plot(timestamps_num, -3*sqrt(z_var), '--')
legend('z error','z 3\sigma','z -3\sigma')
xlabel('Time [s]')
ylabel('Error [m]')
hold off

% Error plot
figure(9)
plot(timestamps_num, roll_error)
hold on
plot(timestamps_num, 3*sqrt(roll_var), '--')
hold on
plot(timestamps_num, -3*sqrt(roll_var), '--')
legend('roll error','roll 3\sigma','roll -3\sigma')
xlabel('Time [s]')
ylabel('Error [m]')
hold off

% Error plot
figure(10)
plot(timestamps_num, pitch_error)
hold on
plot(timestamps_num, 3*sqrt(pitch_var), '--')
hold on
plot(timestamps_num, -3*sqrt(pitch_var), '--')
legend('pitch error','pitch 3\sigma','pitch -3\sigma')
xlabel('Time [s]')
ylabel('Error [m]')
hold off

% Error plot
figure(11)
plot(timestamps_num, yaw_error)
hold on
plot(timestamps_num, 3*sqrt(yaw_var), '--')
hold on
plot(timestamps_num, -3*sqrt(yaw_var), '--')
legend('yaw error','yaw 3\sigma','yaw -3\sigma')
xlabel('Time [s]')
ylabel('Error [m]')
hold off