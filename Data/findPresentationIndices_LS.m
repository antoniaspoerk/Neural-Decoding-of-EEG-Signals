function [labels_final,indices_final,objectName_final]=findPresentationIndices_LS(allCateg, conceptT, neweventlist)


fieldNames = fieldnames(allCateg);
for fieldn = 1:length(fieldNames)
    n(fieldn,1) = size(allCateg.(fieldNames{fieldn}),1);
end


strings1=conceptT.Word; % stimuli names
strings2=neweventlist.object; % Object presented in each trial
labels_final = [];
indices_final = [];
objectName_final = [];

for numcat = 1:length(fieldNames)
    k=1; l=0;labels1=[]; indices1=[]; eegInd=[]; objectOut1=[];objectName1={};
    catloc = allCateg.(fieldNames{numcat});
    nx = n(numcat);
    for i=1:nx

        % find locations in EEG trials of each animate object
        % Finds the location of strings1 within strings2
        eegInd{i}=find(strcmp(strings1(catloc(i)),strings2));

        if(~isempty(eegInd{i}))
            l=l+length(eegInd{i});
            labels1(k:l,1)=i;
            indices1(k:l,1)=eegInd{i};
            k=k+length(eegInd{i});
            objectName1(i)=strings1(catloc(i));
        else
            objectOut1(i)=1;
        end
    end
    labels1(:,2)=numcat; %% animate label
    labels_final = [labels_final;labels1];
    indices_final = [indices_final;indices1];
    objectName_final = [objectName_final,objectName1];
end




