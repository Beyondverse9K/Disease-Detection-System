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
        c3 = [];  % Initialize c3 for reduced training set
        c4 = [];  % Initialize c4 for reduced group
        j = 1;
        k = 1;

        if N > 1
            itr = 1;
            classes = 0;
            cond = max(C) - min(C);

            while (classes ~= 1) && (itr <= length(u)) && (size(C, 1) > 0) && (cond > 0)
                c1 = (C == u(itr));
                newClass = c1;

                % Train the SVM model using fitcsvm
                svmModel = fitcsvm(T, newClass, 'KernelFunction', 'rbf');  % RBF kernel
                classes = predict(svmModel, tst);

                % Reduction of Training Set
                for i = 1:length(newClass)
                    if newClass(i) == 0
                        c3(k, :) = T(i, :);
                        k = k + 1;
                    end
                end
                T = c3;  % Update T
                c3 = []; % Reset c3
                k = 1;   % Reset k

                % Reduction of Group
                for i = 1:length(newClass)
                    if newClass(i) == 0
                        c4(j) = C(i);
                        j = j + 1;
                    end
                end
                C = c4;  % Update C
                c4 = []; % Reset c4
                j = 1;   % Reset j

                cond = max(C) - min(C);  % Update condition

                % Increment iteration index if classes are not yet determined
                if classes ~= 1
                    itr = itr + 1;
                end
                if itr > length(u)
                    break;  % Exit if itr exceeds the number of unique classes
                end
            end
        end

        % Logic for classification of multiple rows in the testing matrix
        valt = Cb == u(itr);
        val = Cb(valt == 1);
        val = unique(val);
        itrfin(tempind) = val;  % Store the result for the current test sample
    end
end