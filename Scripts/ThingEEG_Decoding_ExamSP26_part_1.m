% Data are available at:
%https://www.dropbox.com/scl/fi/nnugug12ftjp2sfdlo4r8/Data_Week5.zip?rlkey=6j0rpl91jssqtq13ju4custlb&st=xk3nfu2u&dl=0
%%
clear all

load('S0_epoched.mat')
load('S0_epoched_eventlist.mat')

% Load object concept info: Bottom_upCategory_HumanRaters_ is how humans
% actually categorized each stimulus. Decoding is done category-wise,
% not stimulus-wise.
T=readtable('object_concepts.xlsx');

rng(40);
clearvars -except EEG_epoch neweventlist T DA_sh
times=EEG_epoch.times;

%% Get trials of interest
% EEG data is reformatted to trials x electrodes x time
data2=permute(EEG_epoch.data,[3 1 2]);

% Animal classes: animal, insect, bird
animalLocs=find(strcmp(T.Bottom_upCategory_HumanRaters_,'animal'));
insectLocs = find(strcmp(T.Bottom_upCategory_HumanRaters_,'insect'));
birdLocs=find(strcmp(T.Bottom_upCategory_HumanRaters_,'bird'));

% Non-animal classes: tool, fruit, vegetable, vehicle
toolLocs = find(strcmp(T.Bottom_upCategory_HumanRaters_,'tool'));
fruitLocs = find(strcmp(T.Bottom_upCategory_HumanRaters_,'fruit'));
vegLocs = find(strcmp(T.Bottom_upCategory_HumanRaters_,'vegetable'));
vehicleLocs = find(strcmp(T.Bottom_upCategory_HumanRaters_,'vehicle'));

allCateg = struct();
allCateg.animalClass=unique([animalLocs;insectLocs;birdLocs]);
allCateg.nonanimalClass=unique([toolLocs;fruitLocs;vegLocs;vehicleLocs]);

% labelsOut2(:,2): category per trial (1 = animalClass, 2 = nonanimalClass)
% indices2: trials of interest (animal or non-animal images)
% objectNames2: specific images considered
[labelsOut2,indices2,objectNames2]=findPresentationIndices_LS(allCateg, T, neweventlist);

categoryLabels = {'animalClass','nonanimalClass'};
data=data2(indices2,:,:);
size(data)

[dataCell{1:2,2}]=deal('animalClass','nonanimalClass');
for i_conds = 1:2
    flagTrials = ismember(labelsOut2(:,2),i_conds);
    dataCell{i_conds,1} = single(data(flagTrials,:,:));
end

%% Equalize trial counts across categories
clear newdataCell
coi = 1:2;
minC = min(histc(labelsOut2(:,2),coi));

% Reduce the number of trials just for practice
if minC > 300
    minC = 100;
end

[newdataCell{1:2,2}]=deal('animalClass','nonanimalClass');

for cond2eq = 1:length(coi)
    origdata = dataCell{coi(cond2eq),1};
    subdata = origdata(randperm(size(origdata,1),minC),:,:);
    newdataCell{cond2eq,1} = subdata;
end

%% Parameters
iterationX = 10;       % repetitions of the training/testing procedure
conditionM = 2;

trials_pseudo=5;                       % raw trials averaged per pseudo trial
n_pseudo=floor(minC/trials_pseudo);    % pseudo trials per class

timeSamps = [1:2:length(times)]';      % downsample: test every other time point
nTS = length(timeSamps);

n_folds=5;

DA = nan(iterationX, conditionM, conditionM, nTS);

for iterX = 1:iterationX

    % Reshape into conditions x pseudo-trials x electrodes x timepoints,
    % averaging groups of shuffled raw trials into each pseudo trial
    clear pseudoData
    for condIndex = 1:conditionM
        conditionMatrix = newdataCell{condIndex, 1};
        rawTrialNumber = size(conditionMatrix, 1);
        shuffled=conditionMatrix(randperm(rawTrialNumber), :, :);
        for p=1:n_pseudo
            binIdx =(p-1)*trials_pseudo+(1:trials_pseudo);
            pseudoData(condIndex,p,:,:) = mean(shuffled(binIdx, :, :), 1);
        end
    end

    cv = cvpartition(n_pseudo,"kfold",n_folds);
    acc_folds =nan(n_folds,nTS);

    for fold = 1:n_folds
        traintrial_idx = training(cv,fold);
        testtrial_idx = test(cv,fold);
        % nchoosek gives all condition pairs (useful when conditionM > 2)
        for pair = nchoosek(1:conditionM, 2)'
            condA = pair(1);
            condB = pair(2);
            for timeT = 1:nTS
                trainingdataA = double(squeeze(pseudoData(condA,traintrial_idx,:,timeSamps(timeT))));
                trainingdataB = double(squeeze(pseudoData(condB,traintrial_idx,:,timeSamps(timeT))));
                trainingdata = [trainingdataA;trainingdataB];

                testingdataA = double(squeeze(pseudoData(condA,testtrial_idx, :, timeSamps(timeT))));
                testingdataB = double(squeeze(pseudoData(condB, testtrial_idx, :, timeSamps(timeT))));
                testingdata = [testingdataA;testingdataB];

                % Normalize using training-set mean/SD, applied to both sets
                mu=mean(trainingdata,1);
                sd=std(trainingdata,0,1);
                sd(sd==0)=1; % avoid division by zero for constant electrodes
                trainingdata=(trainingdata-mu)./sd;
                testingdata=(testingdata-mu)./sd;

                labels_train = [ones(size(trainingdataA,1),1);2*ones(size(trainingdataB,1),1)];
                model = fitcsvm(trainingdata, labels_train, 'KernelFunction', 'linear', 'BoxConstraint', 1);

                labels_test = [ones(size(testingdataA,1),1);2*ones(size(testingdataB,1),1)];
                [predicted_labels, scores] = predict(model, testingdata);
                accuracy = mean(predicted_labels == labels_test) * 100;

                acc_folds(fold,timeT)=accuracy;
            end
        end
    end

    DA(iterX, condB, condA, :) = mean(acc_folds,1);
    fprintf('Iteration %d done\n', iterX);
end

%% Time course of decoding accuracy
DA_spec = squeeze(DA(:,2,1,:));
accuracyTime = squeeze(nanmean(DA_spec,1));

%% Null distribution via label permutation (chance-level accuracy)
n_perm = 10;
DA_perm = nan(n_perm, nTS);

for permX = 1:n_perm

    acc_perm_iters = nan(iterationX, nTS);

    for iterX = 1:iterationX
        clear pseudoData
        for condIndex = 1:conditionM
            conditionMatrix = newdataCell{condIndex, 1};
            rawTrialNumber  = size(conditionMatrix, 1);
            shuffled = conditionMatrix(randperm(rawTrialNumber), :, :);
            for p = 1:n_pseudo
                binIdx = (p-1)*trials_pseudo + (1:trials_pseudo);
                pseudoData(condIndex, p, :, :) = mean(shuffled(binIdx, :, :), 1);
            end
        end

        allPseudo  = [squeeze(pseudoData(1,:,:,:)); squeeze(pseudoData(2,:,:,:))];
        trueLabels = [ones(n_pseudo,1); 2*ones(n_pseudo,1)];
        permLabels = trueLabels(randperm(length(trueLabels))); % shuffle labels -> chance level

        cv_perm    = cvpartition(length(permLabels), 'KFold', n_folds);
        acc_folds_perm = nan(n_folds, nTS);

        for fold = 1:n_folds
            traintrial_idx = training(cv_perm, fold);
            testtrial_idx  = test(cv_perm, fold);

            for timeT = 1:nTS
                trainData = squeeze(allPseudo(traintrial_idx, :, timeSamps(timeT)));
                testData  = squeeze(allPseudo(testtrial_idx,  :, timeSamps(timeT)));
                trainLbls = permLabels(traintrial_idx);
                testLbls  = permLabels(testtrial_idx);

                mu          = mean(trainData, 1);
                sd          = std(trainData, 0, 1);
                sd(sd == 0) = 1;
                trainData   = (trainData - mu) ./ sd;
                testData    = (testData  - mu) ./ sd;

                model_perm  = fitcsvm(trainData, trainLbls, ...
                    'KernelFunction', 'linear', 'BoxConstraint', 1);
                pred_perm   = predict(model_perm, testData);
                acc_folds_perm(fold, timeT) = mean(pred_perm == testLbls) * 100;
            end
        end
        acc_perm_iters(iterX, :) = mean(acc_folds_perm, 1);
    end
    DA_perm(permX, :) = mean(acc_perm_iters, 1);
    fprintf('Permutation %d/%d done\n', permX, n_perm);
end

permMean   = mean(DA_perm, 1);
permCI_low  = prctile(DA_perm,  2.5, 1);
permCI_high = prctile(DA_perm, 97.5, 1);

%% Plot decoding accuracy vs. permutation null distribution
tPlot = times(timeSamps);
figure;

fill([tPlot, fliplr(tPlot)], [permCI_low, fliplr(permCI_high)], ...
    [0.75 0.75 0.75], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
hold on;
plot(tPlot, permMean, 'k-', 'LineWidth', 1.5);
plot(tPlot, accuracyTime, 'r-', 'LineWidth', 1.5);
plot(times, ones(size(times))*50, 'k--');
xlim([-100, 600]);
xlabel('Time (ms)');
ylabel('Decoding accuracy (%)');
title('Animal vs Non-animal decoding - Subject 0');
legend({'95% CI permutation', 'Permutation mean', 'Decoding accuracy'});

%% Save results
save('DA_part1.mat', 'DA');
save('DA_sh_part1.mat', 'DA_perm');
