function [ labels ] = knn( X, G, T, K )   
    %initialise labels to 0
    labels = cell(size(T,1),1);
    %for all test instances
    for j=1:size(T,1)
        euclidean_distances = zeros(size(X,1), 1);
        %for all train instances
        for i=1:size(X,1)
            %calculate the distance between each attribute
            distanceVector = (X(i,:)-T(j,:)).^2;
            %sum all the distances together
            distance = sum(distanceVector);
            %get the square root
            distance = sqrt(distance);
            %add the distance to the euclidean distance vector
            euclidean_distances(i)= distance; 
        end
        %sort euclidean distance vector from smallest to largest and store
        %the original indecies
        [sorted_euclidean_distances, indecies] = sort(euclidean_distances);
        knn = cell(K,1);
        %for the k nearest points get their class
        for i=1:K
            knn(i) = G(indecies(i));
        end
        %add the mode label to labels
        [unique_strings, ~, string_map]=unique(knn);
        labels(j) = unique_strings(mode(string_map));
    end
end

