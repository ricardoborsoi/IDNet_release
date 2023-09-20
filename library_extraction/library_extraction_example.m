


% load Yim : nr * nc * L  hyperspectral image cube
load '../DATA/real_Samson/alldata_real_Samson.mat'
Y = reshape(Yim, [size(Yim,1)*size(Yim,2), size(Yim,3)])'; % reorder into bands * pixels

num_EMs = 3;

% extract reference EMs using VCA if desired
% M0 = vca(Y,'Endmembers',num_EMs); 


% Extract bundles from the image based on angles (see the description in the IDNet paper) -----------
flag_Npx = true;
vec_Npx = 100*ones(num_EMs,1); % number of pure pixel to extract per endmember
[bundleLibs,avg_M,PPidx,EM_pix,IDX_comp] = extract_bundles_by_angle(Yim, M0, vec_Npx, flag_Npx);


% Extract bundles from the image based on the batch VCA alg. -----------
percent = 0.25;
bundle_nbr = 100;
[bundleLibs2]=extractbundles_batchVCA(Y, M0, bundle_nbr, percent);


save('extracted_bundle_test.mat','bundleLibs')


