%% Project Title: Disease Detection

while true
    choice = menu('Disease Detection', '....... Training........', '....... Testing......', '........ Close........');
    
    if choice == 1
        %% Image Read for Training
        Train_Feat = [];  % Initialize feature matrix
        Train_Label = []; % Initialize label vector
        xx = 1;           % Initialize class label
        
        for k = 1:18
            % Read and preprocess the image
            basePath = 'C:/Users/paulj/OneDrive/Documents/Disease-Detection/Train/';
            filePath = fullfile(basePath, sprintf('Train (%d).jpg', k));
            I = imread(filePath);
            I = imresize(I, [1000, 260]);
            [I3, RGB] = createMask(I);
            seg_img = RGB;
            img = rgb2gray(seg_img);
            glcms = graycomatrix(img);
            
            % Derive statistics from GLCM
            stats = graycoprops(glcms, 'Contrast Correlation Energy Homogeneity');
            Contrast = stats.Contrast;
            Energy = stats.Energy;
            Homogeneity = stats.Homogeneity;
            Mean = mean2(seg_img);
            Standard_Deviation = std2(seg_img);
            Entropy = entropy(seg_img);
            RMS = mean2(rms(seg_img));
            Variance = mean2(var(double(seg_img)));
            a = sum(double(seg_img(:)));
            Smoothness = 1 - (1 / (1 + a));
            
            % Inverse Difference Moment
            m = size(seg_img, 1);
            n = size(seg_img, 2);
            in_diff = 0;
            for i = 1:m
                for j = 1:n
                    temp = seg_img(i, j) / (1 + (i - j)^2);
                    in_diff = in_diff + temp;
                end
            end
            IDM = double(in_diff);

            % Combine features into a single row
            ff = [Contrast, Energy, Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM];
            Train_Feat = [Train_Feat; ff];  % Append to the feature matrix
            
            % Update labels
            if k < 10
                Train_Label = [Train_Label; 1];  % Label for class 1
            else
                Train_Label = [Train_Label; 2];  % Label for class 2
            end
        end
        disp('Training Complete');
    end
    
    if choice == 2
        %% Image Read for Testing
        [filename, pathname] = uigetfile({'*.*'; '*.bmp'; '*.jpg'; '*.gif'}, 'Pick a Leaf Image File');
        if isequal(filename, 0) || isequal(pathname, 0)
            disp('User  canceled the operation.');
            continue;  % Go back to the menu if no file is selected
        end
        
        I = imread(fullfile(pathname, filename));
        I = imresize(I, [1000, 260]);
        figure, imshow(I); title('Query Leaf Image');
        
        %% Create Mask or Segmentation Image
        [I3, RGB] = createMask(I);
        seg_img = RGB;
        figure, imshow(I3); title('BW Image');
        figure, imshow(seg_img); title('Segmented Image');
        
        %% Feature Extraction
        img = rgb2gray(seg_img);
        glcms = graycomatrix(img);
        stats = graycoprops(glcms, 'Contrast Correlation Energy Homogeneity');
        
        % Extract features
        Contrast = stats.Contrast;
        Energy = stats.Energy;
        Homogeneity = stats.Homogeneity;
        Mean = mean2(seg_img);
        Standard_Deviation = std2(seg_img);
        Entropy = entropy(seg_img);
        RMS = mean2(rms(seg_img));
        Variance = mean2(var(double(seg_img)));
        a = sum(double(seg_img(:)));
        Smoothness = 1 - (1 / (1 + a));
        
        % Inverse Difference Moment
        m = size(seg_img, 1);
        n = size(seg_img, 2);
        in_diff = 0;
        for i = 1:m
            for j = 1:n
                temp = seg_img(i,j)./(1+(i-j).^2);
                in_diff = in_diff+temp;
            end
        end
        IDM = double(in_diff);

        feat_disease = [Contrast,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM];
        %% Plot Histogram of Grayscale Image
        figure;
        histogram(glcms(:), 256);  % Use img(:) to flatten the image matrix
        title('Histogram of Grayscale Image');
        xlabel('Pixel Intensity');
        ylabel('Frequency');
        xlim([0 255]);  % Set x-axis limits for 8-bit grayscale images
        grid on;  % Add grid for better visibility
        %% SVM Classifier
        % Load All The Features
        %load('Training_Data.mat')

        % Put the test features into variable 'test'
        test = feat_disease;
        result = multisvm(Train_Feat,Train_Label,test);
        %disp(result);

        
        %% Visualize Results
        if result == 1
            helpdlg(' Disease Detect ');
            disp(' Disease Detect ');
        else
            helpdlg(' Disease not Detect ');
            disp('Disease not Detect');
        end
    end
    if (choice==3)
        close all;
        return;
    end
end