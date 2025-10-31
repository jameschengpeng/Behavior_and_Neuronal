%% plot an event from AQuA2
function evt_trace = get_evt_trace(video4D, evt1, evt_num)
[H, W, ~, T] = size(video4D);
evt_voxel = evt1{evt_num};
[evt_x, evt_y, ~, evt_t] = ind2sub(size(video4D), evt_voxel);
ind = find(evt_t == mode(evt_t));
x_selected = evt_x(ind);
y_selected = evt_y(ind);
z_selected = ones(size(ind));

pixel_one_frame = sub2ind([H W 1], x_selected, y_selected, z_selected);
V = reshape(video4D, [], T); % reshape to (H*W*1) * T
selected = V(pixel_one_frame, :);
evt_trace = mean(selected, 1);
end
