%% Project Title: Disease Detection

while true
    choice = menu('Disease Detection', '....... Training........', '....... Testing......', '........ Close........');
    
    if choice == 1
        %% Image Read for Training
        Train_Feat = [];  % Initialize feature matrix
        Train_Label = []; % Initialize label vector
        basePath = 'C:/Users/paulj/OneDrive/Documents/Disease-Detection/Train/';  

        % Define the class labels and their corresponding folder names
        classLabels = {'Healthy', 'Powdery', 'Rust'};

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
                
                % Data Augmentation
                % Randomly apply transformations
                if rand() > 0.5
                    I = imrotate(I, randi([-15, 15]));  % Random rotation
                end
                if rand() > 0.5
                    I = flip(I, 1);  % Random vertical flip
                end
                if rand() > 0.5
                    I = flip(I, 2);  % Random horizontal flip
                end
                if rand() > 0.5
                    % Randomly adjust brightness
                    I = imadjust(I, [], [], rand() * 0.2 + 0.9); % Adjust brightness
                end
                if rand() > 0.5
                    % Add Gaussian noise
                    I = imnoise(I, 'gaussian', 0, 0.01); % Add Gaussian noise
                end
                
                [I3, RGB] = createMask(I);
                seg_img = RGB;
                img = rgb2gray(seg_img);
                
                % Feature Extraction
                % GLCM features
                glcms = graycomatrix(img);
                stats = graycoprops(glcms, 'Contrast Correlation Energy Homogeneity');
                Contrast = stats.Contrast;
                Energy = stats.Energy;
                Homogeneity = stats.Homogeneity;
                
                % Additional features
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
            continue;  % Go back to the menu if no training data is available
        end

        disp('Training Complete');

        %% Hyperparameter Tuning with Grid Search
        % Define the parameter grid for tuning
        kernelFunctions = {'linear', 'rbf', 'polynomial'};
        boxConstraints = logspace(-3, 3, 7);  % Wider range of box constraints
        bestAccuracy = 0;
        bestModel = [];

        % Grid search over all combinations of hyperparameters
        for kf = 1:length(kernelFunctions)
            for bc = boxConstraints
                % Train the SVM model using fitcecoc for multi-class classification
                svmModel = fitcecoc(Train_Feat, Train_Label, ...
                    'Learners', templateSVM('KernelFunction', kernelFunctions{kf}, 'BoxConstraint', bc));

                % Perform cross-validation
                cvModel = crossval(svmModel);
                accuracy = 1 - kfoldLoss(cvModel);  % Calculate accuracy

                % Display the current hyperparameters and accuracy
                disp(['Kernel: ', kernelFunctions{kf}, ', Box Constraint: ', num2str(bc), ', Accuracy: ', num2str(accuracy * 100), '%']);

                % Update best model if current accuracy is better
                if accuracy > bestAccuracy
                    bestAccuracy = accuracy;
                    bestModel = svmModel;
                end
            end
        end

        % Display the best model and its accuracy
        disp(['Best Model Accuracy: ', num2str(bestAccuracy * 100), '%']);

        %% Performance Analysis
        % Split the data into training and testing sets (80% train, 20% test)
        cv = cvpartition(size(Train_Feat, 1), 'HoldOut', 0.2);
        idx = cv.test;

        % Separate to training and test data
        TrainData = Train_Feat(~idx, :);
        TrainLabels = Train_Label(~idx);
        TestData = Train_Feat(idx, :);
        TestLabels = Train_Label(idx);

        % Make predictions on the test set using the best model
        predictedLabels = predict(bestModel, TestData);

        % Calculate performance metrics
        accuracy = sum(predictedLabels == TestLabels) / length(TestLabels);
        confusionMat = confusionmat(TestLabels, predictedLabels);
        precision = diag(confusionMat) ./ sum(confusionMat, 2);  % Precision for each class
        recall = diag(confusionMat) ./ sum(confusionMat, 1)';  % Recall for each class
        f1Score = 2 * (precision .* recall) ./ (precision + recall);  % F1 Score for each class

        % Display performance metrics
        disp(['Final Accuracy: ', num2str(accuracy * 100), '%']);
        disp('Confusion Matrix:');
        disp(confusionMat);
        disp('Precision for each class:');
        disp(precision);
        disp('Recall for each class:');
        disp(recall);
        disp('F1 Score for each class:');
        disp(f1Score);
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
        for j = 1:m
            for l = 1:n
                temp = seg_img(j, l) / (1 + (j - l)^2);
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

        % Call the best model to classify the test features
        predicted_class = predict(bestModel, test_feat);  % Use the trained best model for prediction
        disp(['Predicted Class: ', num2str(predicted_class)]);
    end
    
    if choice == 3
        disp('Closing the application.');
        break;  % Exit the loop to close the application
    end
end