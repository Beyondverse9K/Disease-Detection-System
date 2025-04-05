%% Project Title: Disease Detection

while true
    choice = menu('Disease Detection', '....... Training........', '....... Testing......', '........ Close........');
    
    if choice == 1
    %% Image Read for Training
    Train_Feat = [];  % Initialize feature matrix
    Train_Label = []; % Initialize label vector
    % Define the base path for training images
basePath = 'C:/Users/paulj/OneDrive/Documents/Disease-Detection/Train/';  

% Define the class labels and their corresponding folder names
classLabels = {'Healthy', 'Powdery', 'Rust'};
Train_Feat = [];  % Initialize feature matrix
Train_Label = []; % Initialize label vector

% Loop through each class folder
for i = 1:length(classLabels)
    className = classLabels{i};
    classFolder = fullfile(basePath, className);  % Full path to the class folder
    imageFiles = dir(fullfile(classFolder, '*.jpg'));  % Get all jpg files in the folder

    disp(['Found ', num2str(length(imageFiles)), ' images in ', className, ' folder.']);  % Debugging statement

    % Loop through each image file in the class folder
    for k = 1:length(imageFiles)
        filePath = fullfile(classFolder, imageFiles(k).name);  % Full path to the image
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
        for j = 1:m
            for l = 1:n
                temp = seg_img(j, l) / (1 + (j - l)^2);
                in_diff = in_diff + temp;
            end
        end
        IDM = double(in_diff);

        % Combine features into a single row
        ff = [Contrast, Energy, Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM];
        Train_Feat = [Train_Feat; ff];  % Append to the feature matrix
        
        % Assign label based on the folder name
        Train_Label = [Train_Label; i];  % Use the index as the label (1 for Healthy, 2 for Powdery, 3 for Rust)
    end
end

% Check if training data is available
if isempty(Train_Feat) || isempty(Train_Label)
    disp('Error: No training data available. Please ensure that the training images are correctly labeled and processed.');
    return;  % Exit if no training data is available
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
                temp = seg_img(i, j) / (1 + (i - j)^2);
                in_diff = in_diff + temp;
            end
        end
        IDM = double(in_diff);

        % Combine features into a single row for testing
        test_feat = [Contrast, Energy, Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM];

        % Check if training data is available before classification
        if isempty(Train_Feat) || isempty(Train_Label)
            disp('Error: No training data available. Please train the model first.');
            continue;  % Go back to the menu if no training data
        end

        % Call the multisvm function to classify the test features
        predicted_class = multisvm(Train_Feat, Train_Label, test_feat);
        if predicted_class == 1
            pc="Healthy";
        elseif predicted_class == 2
            pc="Powdery";
        else
            pc="Rust";
        end
        disp('Predicted Class: ', pc);
    end
    
    if choice == 3
        disp('Closing the application.');
        break;  % Exit the loop to close the application
    end
end