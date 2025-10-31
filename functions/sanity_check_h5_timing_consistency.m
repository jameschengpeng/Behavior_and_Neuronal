function sanity_check_h5_timing_consistency(h5_path)
    info = h5info(h5_path);
    num_issues = 0;

    for i = 1:length(info.Groups)
        group_name = info.Groups(i).Name;

        try
            % Read the shape of the video and ethogram
            video_ds = h5read(h5_path, group_name + "/X");
            ethogram_ds = h5read(h5_path, group_name + "/y");

            T_video = size(video_ds, 1);      % (T, H, W, C)
            T_ethogram = size(ethogram_ds, 1);  % (T, C)

            if T_video ~= T_ethogram
                fprintf('[Mismatch] %s: video T = %d, ethogram T = %d\n', group_name, T_video, T_ethogram);
                num_issues = num_issues + 1;
            else
                fprintf('[OK] %s: T = %d\n', group_name, T_video);
            end
        catch ME
            fprintf('[Error] %s: %s\n', group_name, ME.message);
            num_issues = num_issues + 1;
        end
    end

    if num_issues == 0
        fprintf('\n✅ All entries passed the timing consistency check.\n');
    else
        fprintf('\n⚠️  %d issues found. Check output above.\n', num_issues);
    end
end
