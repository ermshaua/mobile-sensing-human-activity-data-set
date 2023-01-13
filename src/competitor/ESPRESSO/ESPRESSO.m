% run ESPRESSO
function espTT = ESPRESSO(data, subsequence, max_cps, chain_length)
%%% Input 
% data : input original data ,
% subsequence : window size
% chain_len : maximum length of chain for each subsequence
%%% returns:
% extracted segment boundaries, F-score, MAE
    [numTS,lenTS] = size(data);    % number of time-series and length of them

    [MP, MPI] = computMP(data, subsequence);

    for i = [1:size(MP,1)]
        [wcac(i,:)] = calculateSemanticDensityMatrix(MP,MPI, chain_length, subsequence);
    end
    [espTT,~] = separateGreedyIG(data, max_cps+1, wcac, 0.01);
end


function [MP, MPI] = computMP(Integ_TS, subsequence)
    for i=1:1:size(Integ_TS,1)
        [MP(i,:), MPI(i,:)] = timeseriesSelfJoinFast(Integ_TS(i,:),subsequence);
    end
end
