function L = build_laplacian_active(H, W, maskVec, neighborhood)
%BUILD_LAPLACIAN_ACTIVE  Build sparse graph Laplacian over active (masked) pixels.
%
% Constructs the combinatorial Laplacian L = D - W over the set of active
% pixels in a H x W image defined by maskVec.  Edges connect spatially
% adjacent active pixels using 4- or 8-connectivity.
%
% Inputs
%   H            : image height (rows)
%   W            : image width  (columns)
%   maskVec      : H*W logical vector (or logical H x W image); nonzero = active
%   neighborhood : 4 (default) or 8
%
% Output
%   L : Pm x Pm sparse symmetric positive-semidefinite Laplacian,
%       where Pm = nnz(maskVec)

    maskVec = reshape(maskVec, [], 1) ~= 0;
    idxFull = find(maskVec);
    Pm = numel(idxFull);

    % Map from full-image linear index to active-pixel index
    map = zeros(H*W, 1, 'int32');
    map(idxFull) = int32(1:Pm);

    % Row/col of every active pixel
    [ri, ci] = ind2sub([H, W], idxFull);

    % Neighbor offsets
    if neighborhood == 8
        offsets = int32([-1 0; 1 0; 0 -1; 0 1; -1 -1; -1 1; 1 -1; 1 1]);
    else
        offsets = int32([-1 0; 1 0; 0 -1; 0 1]);
    end

    nDir = size(offsets, 1);
    src = cell(nDir, 1);
    dst = cell(nDir, 1);
    active_idx = (1:Pm)';

    for d = 1:nDir
        rn = ri + double(offsets(d, 1));
        cn = ci + double(offsets(d, 2));

        valid = (rn >= 1) & (rn <= H) & (cn >= 1) & (cn <= W);

        nb_full = zeros(Pm, 1);
        nb_full(valid) = sub2ind([H, W], rn(valid), cn(valid));

        nb_active = zeros(Pm, 1);
        nb_active(valid) = double(map(nb_full(valid)));

        keep = valid & (nb_active > 0);
        src{d} = active_idx(keep);
        dst{d} = nb_active(keep);
    end

    src_all = vertcat(src{:});
    dst_all = vertcat(dst{:});
    Wmat = sparse(src_all, dst_all, 1, Pm, Pm);

    deg = full(sum(Wmat, 2));
    L = spdiags(deg, 0, Pm, Pm) - Wmat;
end
