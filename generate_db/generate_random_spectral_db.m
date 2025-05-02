% Generate simulated dataset 
% Sub: Example on the spectral domain using ERSP 
% Based on the design from: Ravindran, A.S. & Contreras-Vidal, J. (2023)

% Author: Ade Nurhopipah
% Email: ade.nurhopipah@postgrad.otago.ac.nz

function generate_random_spectral(class, SNR)
    % Check the number of input arguments and set default values if necessary
    if nargin < 2
        SNR = -3.5;  % Default value for SNR
    end
    if nargin < 1
        class = 1;  % Default value for class
    end

    % Display the values of class and SNR
    disp(['Class: ', num2str(class)]);
    disp(['SNR: ', num2str(SNR)]);

    %define leadfield
    %using the New York head model with actiCAP64 channel projection
    lf = lf_generate_fromnyhead('montage', 'actiCAP64');
    
    %difene 14 dipole ERP sources
    dipole_loc = [-40, -21, 51;
                    40, -21, 51;
                    -38, -26, 53;
                    38, -26, 53;
                    -48, -15, 50;
                    48, -15, 50;
                    -24, -24, 32;
                    24, -24, 32;
                    -34, -32, 38;
                    34, -32, 38;
                    -42, 40, 25;
                    42, 40, 25;
                    0, -4, 65;
                    8, -12, 52];

    %adjust SNR 
    if SNR == -3.5
        snr_ratio = 0.4005; 
    elseif SNR == -12
        snr_ratio = 0.2007; 
    elseif SNR == -16
        snr_ratio = 0.1368;
    elseif SNR == -19
        snr_ratio = 0.1009;
    elseif SNR == -23
        snr_ratio = 0.0661;
    end 
    
    %create a structured epoch array
    num_data = 100;
    epochs = struct();
    epochs.n = 100;               
    epochs.srate = 250;     
    epochs.length = 1000; 
    
    %create class in spectral domain
    ersp = struct();
    ersp.type = 'ersp';
    ersp.amplitude = 1.75;
    ersp.amplitudeDv = 1.25;
    ersp.modLatency = 350;
    ersp.modLatencyDv = 150;
    ersp.modWidth = 250;
    ersp.modWidthDv = 50;
    ersp.modulation = 'burst';
    ersp.modTaper = 0.5;  
    
    %pink noise (represents source noise)
    pink_noise = struct();
    pink_noise.type = 'noise';
    pink_noise.color = 'pink';
    pink_noise.amplitude = 5;
    pink_noise = utl_check_class(pink_noise);

    %white noise (represents sensor noise)
    white_noise = struct(); 
    white_noise.type = 'noise';
    white_noise.color = 'white';
    white_noise.amplitude = 5;
    white_noise = utl_check_class(white_noise);
    
    for i = 1  : num_data 
        disp(['Generating data iteration-',num2str(i)]);
        if class == 1                           %tetha
            ersp.frequency = 5.5;
            ersp.frequencyDv = 2.5;
            freq_info = '3-8 Hz';
        elseif class == 2                        %alpha
            ersp.frequency = 10.5;
            ersp.frequencyDv = 2.5;             
            freq_info = '8-13 Hz';
        elseif class == 3                        %beta
            ersp.frequency = 22;
            ersp.frequencyDv = 8;  
            freq_info = '14-30 Hz';
        else                                     %gamma
            ersp.frequency = 44;
            ersp.frequencyDv = 14;
            freq_info = '30-58 Hz';
        end
        
        ersp = utl_check_class(ersp);  

        %create pure eeg from erp
        random_idx = randi([1, 14]);    
        random_loc = dipole_loc(random_idx, :);         
        source = lf_get_source_nearest(lf, random_loc);
    
        %create erp component 
        pure_comp = struct();
        pure_comp.source = source;
        pure_comp.signal = {ersp} ;
        
        %project eeg source with erp 
        pure_scalpdata = generate_scalpdata(pure_comp, lf, epochs);
                  
        %create external noise component 
        noise_comp = struct();
        noise_comp.source = source;
        noise_comp.signal = {pink_noise, white_noise};   
        noise_comp = utl_check_component(noise_comp, lf);
       
        %project noise to the source
        noise_scalpdata = generate_scalpdata(noise_comp, lf, epochs);      
    
        %add noise to signal
        eeg_snr_ratio = utl_mix_data(pure_scalpdata, noise_scalpdata, snr_ratio); 
        simulated_eeg = utl_create_eeglabdataset(eeg_snr_ratio, epochs, lf);
        pure_eeg =  utl_create_eeglabdataset(pure_scalpdata, epochs, lf);
        
        if i == 1
            eeg_db = simulated_eeg;
            gt_eeg = pure_eeg;
        else
            eeg_db = pop_mergeset(eeg_db, simulated_eeg);
            gt_eeg = pop_mergeset(gt_eeg, pure_eeg);
        end

    end
        
    % Save the EEG data to the specified path
    gt_name = ['gt_db_spectral', num2str(class),'.set'];
    gt_path = ['H:\serega_db\spectral\class', num2str(class), '\'];
    pop_saveset(gt_eeg, 'filename', gt_name, 'filepath', gt_path); 

    db_name = ['sim_db_spectral', num2str(class),'.set'];
    db_path = ['H:\serega_db\spectral\class', num2str(class), '\'];
    pop_saveset(eeg_db, 'filename', db_name, 'filepath', db_path);  
        
    %save info in .txt
    info_file =  sprintf('H:\\serega_db\\spectral\\class%d\\info_spectral%d.txt', class, class);
    fileID = fopen(info_file, 'w');
    fprintf(fileID, 'EEG synthetic data from SEREGA\n');
    fprintf(fileID, '\nLeadfield : NewYork head model (actiCAP64 projection)'); 
    fprintf(fileID, '\nSpectral class : %s\n', num2str(class)); 
    fprintf(fileID, '\nEpochs info\n'); 
    fprintf(fileID, 'N-epochs : %d x %d\n', num_data, epochs.n);
    fprintf(fileID, 'Sampling rate : %d\n', epochs.srate);
    fprintf(fileID, 'Length of epoch : %d\n', epochs.length);
    fprintf(fileID, '\nERSP info\n'); 
    fprintf(fileID, 'ERSP peak amplitude : %.2f\n', ersp.amplitude); 
    fprintf(fileID, 'ERSP peak amplitude deviation : %.2f\n', ersp.amplitudeDv); 
    fprintf(fileID, 'ERSP peak latency : %d\n', ersp.modLatency); 
    fprintf(fileID, 'ERSP peak latency deviation : %d\n', ersp.modLatencyDv); 
    fprintf(fileID, 'ERSP peak width : %d\n', ersp.modWidth); 
    fprintf(fileID, 'ERSP peak width : %d\n', ersp.modWidthDv); 
    fprintf(fileID, 'ERSP brust modulation with mod taper :  %.2f\n', ersp.modTaper);
    fprintf(fileID, 'ERSP band frequency : %s \n', freq_info);
    fprintf(fileID, '\nSource info:'); 
    for i = 1:14
        loc_source = dipole_loc(i, :);  
        source_class = lf_get_source_nearest(lf, loc_source);
        fprintf(fileID, '\nCoordinates: %.2f, %.2f, %.2f ', loc_source(1), loc_source(2), loc_source(3));
        fprintf(fileID, '(%d)', source_class );
    end 
    fprintf(fileID, '\n\nPink and white noise added with amplitude: 5 microvolt'); % and deviation 3 microvolt'); 
    fprintf(fileID, '\nSNR : %s\n', num2str(SNR)); 
    fclose(fileID);

    % Create and save sample plots
    plot_source_location(source, lf);             
    saveas(gcf, ['H:\serega_db\spectral\class', num2str(class), '\2dsource_spectral', num2str(class), '.png']);  
    plot_source_location(source, lf, 'mode', '3d');           
    saveas(gcf, ['H:\serega_db\spectral\class', num2str(class), '\3dsource_spectral', num2str(class), '.png']); 
    plot_source_projection(source, lf);           
    saveas(gcf, ['H:\serega_db\spectral\class', num2str(class), '\projection_spectral', num2str(class), '.png']);  
    plot_signal_fromclass(ersp, epochs); 
    saveas(gcf, ['H:\serega_db\spectral\class', num2str(class), '\ersp_spectral', num2str(class), '.png']); 
    pop_eegplot(simulated_eeg, 1, 1, 1); 
    saveas(gcf, ['H:\serega_db\spectral\class', num2str(class), '\db_plot_spectral', num2str(class), '.png']);
    pop_eegplot(pure_eeg, 1, 1, 1); 
    saveas(gcf, ['H:\serega_db\spectral\class', num2str(class), '\ground_truth_spectral', num2str(class), '.png']);
         
    disp(['All result saved at: H:\serega_db\spectral\class',num2str(class)]);
  
end

SNR = -3.5;
classes = [1,2,3,4];
for class_id = classes
    generate_random_spectral(class_id, SNR);
end
