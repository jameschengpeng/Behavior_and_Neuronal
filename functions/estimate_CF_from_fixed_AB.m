function [C, F] = estimate_CF_from_fixed_AB(X, A, B)
% Estimate temporal components C and F given fixed spatial A and B.
%
% X: (P x T) data matrix for day 28 (P pixels, T time)
% A: (P x K) neuron spatial footprints from day 21
% B: (P x Kb) background spatial components from day 21
%
% Returns:
% C: (K x T)
% F: (Kb x T)

    P = size(X,1);
    T = size(X,2);
    assert(size(A,1) == P, 'A and X must have same #pixels (rows).');
    assert(size(B,1) == P, 'B and X must have same #pixels (rows).');

    K  = size(A,2);
    Kb = size(B,2);

    D = [A, B];           % (P x (K+Kb))
    C = zeros(K,  T);
    F = zeros(Kb, T);

    % Solve independently per time point
    for t = 1:T
        y = X(:,t);
        z = lsqnonneg(D, y);   % z >= 0
        C(:,t) = z(1:K);
        F(:,t) = z(K+1:end);
    end
end
