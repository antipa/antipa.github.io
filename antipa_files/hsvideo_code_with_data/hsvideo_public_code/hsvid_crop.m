function indicator = hsvid_crop(dims, nlines, pad2d)
    indicator = zeros(dims,'like', false);
    counter = 0;
    layer = ones(dims,'like', true);

    while sum(layer(:))
        layer = zeros(dims);%,'like', false);
        counter = counter+1;
        top_min = max(1,counter-nlines+1);
        top_max = min(counter, floor(dims(1)/2));
        bot_min = max(dims(1)-counter+1, floor(dims(1)/2)+1);
        bot_max = min(dims(1), 1+dims(1)-counter + nlines-1);

        layer(top_min:top_max,:) = true;
        layer(bot_min:bot_max,:) = true;

        if counter == 1
            indicator = pad2d(layer);
        elseif sum(layer(:)) % If empty, don't keep
            indicator = cat(3,indicator,pad2d(layer));
        end
    if counter > dims(1)
        error('while loop is going on too long in hsvid_crop.m')
        break
    end
    end
    
    indicator = indicator(:,:,1:floor(size(indicator,3)/2)*2);
end