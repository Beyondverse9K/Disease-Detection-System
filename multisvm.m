function [itrfin] = multisvm(T, C, test)
    % Inputs: T = Training Matrix, C = Group, test = Testing matrix
    % Outputs: itrfin = Resultant class

    itrind = size(test, 1);
    itrfin = zeros(itrind, 1);  % Preallocate output for efficiency
    Cb = C;
    Tb = T;

    for tempind = 1:itrind
        tst = test(tempind, :);
        C = Cb;
        T = Tb;
        u = unique(C);
        N = length(u);
        
        if N <= 1
            % If there's only one class, assign it directly
            itrfin(tempind) = C(1);
            continue;
        end

        itr = 1;  % Initialize itr
        classes = 0;

        while (classes ~= 1) && (itr <= length(u))
            c1 = (C == u(itr));  % Create a binary class vector
            newClass = double(c1);  % Convert logical to double for SVM

            % Train the SVM model using fitcsvm
            svmModel = fitcsvm(T, newClass, 'KernelFunction', 'rbf');  % RBF kernel
            classes = predict(svmModel, tst);

            % Check if the prediction is valid
            if classes == 1
                itrfin(tempind) = u(itr);  % Store the result for the current test sample
                break;  % Exit the loop if a class is found
            end

            itr = itr + 1;  % Increment iteration index
        end

        % If no valid class was found, assign NaN
        if itr > length(u)
            itrfin(tempind) = NaN;  % Assign NaN if no valid class is found
        end
    end
end