function [top, bottom, left, right] = bboxFromHomography(moving, H)
    % compute valid region from homogrphy H
    
    [h, w, c] = size(moving);
    corners = [1, 1;  % left-top
               1, h;  % left-bottom
               w, h;  % right-bottom
               w, 1]; % right-top

    wc = tformfwd(maketform('projective', double(H)), corners);

    top = max(ceil(max(wc(1,2), wc(4,2))),1);
    bottom = min(floor(min(wc(2,2), wc(3,2))),h);

    left = max(ceil(max(wc(1,1), wc(2,1))),1);
    right = min(floor(min(wc(3,1), wc(4,1))),w);
        
    
end

