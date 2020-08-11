% BM3DProfile: Specify parameters for BM3D
classdef BM3DProfile
    properties (Constant)
        NO_VALUE = -1;
        HARD_THRESHOLDING = int8(1);
        WIENER_FILTERING = int8(2);
        ALL_STAGES = BM3DProfile.HARD_THRESHOLDING + BM3DProfile.WIENER_FILTERING;
    end
    properties
        
         %%%% Select transforms ('dct', 'dst', 'hadamard', or anything that is listed by 'help wfilters'):
        transform_2D_HT_name     = 'bior1.5';%'dct';% %% transform used for the HT filt. of size N1 x N1
        transform_2D_Wiener_name = 'dct';     %% transform used for the Wiener filt. of size N1_wiener x N1_wiener
        transform_3rd_dim_name   = 'haar';%'dct';%    %% transform used in the 3-rd dim, the same for HT and Wiener filt.

        % -- Exact variances for correlated noise: --

        % Variance calculation parameters
        Nf = 32;  % domain size for FFT computations
        Kin = 4;  % how many layers of var3D to calculate

        denoise_residual = false; % Perform residual thresholding and re-denoising
        residual_thr = 3; % Threshold for the residual HT (times sqrt(PSD))
        max_pad_size = [-1, -1]; % Maximum required pad size (= half of the kernel size), or [-1, -1] -> use image size

        % Block matching
        gamma = 4.0;
        
        print_info = false;
        
        %%%% Hard-thresholding (HT) parameters:
        N1                  = 8;%   %% N1 x N1 is the block size used for the hard-thresholding (HT) filtering
        Nstep               = 3;   %% sliding step to process every next reference block
        N2                  = 16;  %% maximum number of similar blocks (maximum size of the 3rd dimension of a 3D array)
        Ns                  = 39;  %% length of the side of the search neighborhood for full-search block-matching (BM), must be odd
        tau_match           = 3000;%% threshold for the block-distance (d-distance)
        beta                = 2.0; %% parameter of the 2D Kaiser window used in the reconstruction
        
        
        
        % threshold parameter for the hard-thresholding in 3D transform domain
        % NO_VALUE in lambda or lamdba_wiener means automatic selection.
        
        lambda_thr3D        = BM3DProfile.NO_VALUE; %2.7;
        % Refiltering
        lambda_thr3D_re       = BM3DProfile.NO_VALUE;
        
        %%%% Wiener filtering parameters:
        N1_wiener           = 8;%4;%
        Nstep_wiener        = 3;
        N2_wiener           = 32;%8;%
        Ns_wiener           = 39;
        tau_match_wiener    = 400;
        beta_wiener         = 2.0;
        
        mu2       = BM3DProfile.NO_VALUE %1.0;
        mu2_re      = BM3DProfile.NO_VALUE;
        
        decLevel = 0;        %% dec. levels of the dyadic wavelet 2D transform for blocks (0 means full decomposition, higher values decrease the dec. number)
    
    end
    methods
        
        function pro = BM3DProfile(profileName)
            
            if ~exist('profileName','var')
                profileName = 'np'; %% default profile
            end
            
            if strcmp(profileName, 'np') == 1
            elseif strcmp(profileName, 'refilter') == 1
                pro.denoise_residual = true;
                
            elseif strcmp(profileName, 'lc') == 1
                
                pro.Nstep               = 6;
                pro.Ns                  = 25;
                pro.Nstep_wiener        = 5;
                pro.N2_wiener           = 16;
                pro.Ns_wiener           = 25;
                
            % Profile 'vn' was proposed in
            %  Y. Hou, C. Zhao, D. Yang, and Y. Cheng, 'Comment on "Image Denoising by Sparse 3D Transform-Domain
            %  Collaborative Filtering"', accepted for publication, IEEE Trans. on Image Processing, July, 2010.
            % as a better alternative to that initially proposed in [1] (which is currently in profile 'vn_old')
            elseif strcmp(profileName, 'vn') == 1
                
                pro.N2                  = 32;
                pro.Nstep               = 4;
                
                pro.N1_wiener           = 11;
                pro.Nstep_wiener        = 6;
                
                pro.lambda_thr3D        = 2.8;
                pro.tau_match_wiener    = 3500;
                
                pro.Ns_wiener           = 39;
                
                
                
             % The 'vn_old' profile corresponds to the original parameters for strong noise proposed in [1].
            elseif strcmp(profileName, 'vn_old') == 1
                
                pro.transform_2D_HT_name = 'dct';
                
                pro.N1                  = 12;
                pro.Nstep               = 4;
                
                pro.N1_wiener           = 11;
                pro.Nstep_wiener        = 6;
            
                pro.lambda_thr3D        = 2.8;
                pro.tau_match_wiener    = 3500;
                pro.tau_match           = 5000;                
                pro.Ns_wiener           = 39;
                                
                
            elseif strcmp(profileName, 'high') == 1 %% this profile is not documented in [1]
                
                pro.decLevel     = 1;
                pro.Nstep        = 2;
                pro.Nstep_wiener = 2;
                pro.lambda_thr3D = 2.5;
                pro.beta         = 2.5;
                pro.beta_wiener  = 1.5;
            elseif strcmp(profileName, 'deb') == 1
                % Parameters for deblurring    
                pro.transform_2D_HT_name = 'dst';
                pro.lambda_thr3D         = 2.9;

                pro.N1_wiener           = 8;
                pro.Nstep_wiener        = 2;
                pro.N2_wiener           = 16;
                pro.Ns_wiener           = 39;
                pro.tau_match_wiener    = 800;
                pro.beta_wiener         = 0;     
                    
            else
                disp('Error: profile not found! Returning default profile.')
            end
            
        end
    end
end




