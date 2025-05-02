% Generate simulated dataset 
% Sub: Example on the temporal domain using ERP 
% Based on the design from: Ravindran, A.S. & Contreras-Vidal, J. (2023)

% Author: Ade Nurhopipah
% Email: ade.nurhopipah@postgrad.otago.ac.nz

function generate_random_temporal(class, SNR) 
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

    % define leadfield: New York head model with actiCAP64 channel projection
    lf = lf_generate_fromnyhead('montage', 'actiCAP64');
    
    %define 10 dipole ERP sources
    dipole_loc = [-9.1, -8.5, 60.2;
                  10.1, -6.9, 62.3;
                  4.6, -3.4, 54.3;
                  8.4, -9.9, 57.9;
                  7.5, -1.6, 53.5;
                  -2.6, -33.9, 54.5;
                  -3.5, -30.7, 52.1;
                  -4.1, -43.2, 49.7;
                  -3.6, -39.2, 46.1;
                  -3.3, -26, 50.4];

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
    
    %Create ERP
    erp = struct();
    if class == 1                           %positive deflection
        erp.peakLatency = 60; 
        erp.peakLatencyDv = 8;
        erp.peakWidth = 50;  
        erp.peakWidthDv = 2;
        erp.peakAmplitude = 7; 
        erp.peakAmplitudeDv = 6;
        dipole_class = 1:5;
    elseif class == 2                       %different latency 
        erp.peakLatency = 900; 
        erp.peakLatencyDv = 5;
        erp.peakWidth = 100;  
        erp.peakWidthDv = 4;
        erp.peakAmplitude = 7; 
        erp.peakAmplitudeDv =6;
        dipole_class = 1:5;
    elseif class == 3                       %different sign (negative)   
        erp.peakLatency = 500; 
        erp.peakLatencyDv = 8;
        erp.peakWidth = 100;  
        erp.peakWidthDv = 4;
        erp.peakAmplitude = -7; 
        erp.peakAmplitudeDv =6;
        dipole_class = 1:5;
    elseif class == 4                       %different source position   
        erp.peakLatency = 500; 
        erp.peakLatencyDv = 8;
        erp.peakWidth = 100;  
        erp.peakWidthDv = 4;
        erp.peakAmplitude = -7; 
        erp.peakAmplitudeDv =6;
        dipole_class = 6:10;
    end
    erp = utl_check_class(erp, 'type', 'erp');
      
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
        %create a class in the temporal domain
        
        if class == 1                           %positive deflection
            random_idx = randi([1, 5]);             
        elseif class == 2                       %different latency
            random_idx = randi([1, 5]); 
        elseif class == 3                       %different sign (negative)
            random_idx = randi([1, 5]);    
        else                                    %different source position
            random_idx = randi([6, 10]);    
        end
        
        %create pure eeg from erp
        random_loc = dipole_loc(random_idx, :);         
        source = lf_get_source_nearest(lf, random_loc);

        %fprintf(file_id, 'Random location %s : %d\n',  num2str(i), source);
    
        %create erp component 
        pure_comp = struct();
        pure_comp.source = source;
        pure_comp.signal = {erp} ;
        
        %project eeg source with erp 
        pure_scalpdata = generate_scalpdata(pure_comp, lf, epochs);
                  
        %create noise component 
        noise_comp = struct();
        noise_comp.source = source;
        noise_comp.signal = {pink_noise, white_noise};   
        noise_comp = utl_check_component(noise_comp, lf);
       
        %project noise to the source
        noise_scalpdata = generate_scalpdata(noise_comp, lf, epochs);      
    
        %add noise to signal
        EEG_snr_ratio = utl_mix_data(pure_scalpdata, noise_scalpdata, snr_ratio); 
        pure_eeg =  utl_create_eeglabdataset(pure_scalpdata, epochs, lf);
        simulated_eeg = utl_create_eeglabdataset(EEG_snr_ratio, epochs, lf);
        
        %merge dataset
        if i == 1
            eeg_db = simulated_eeg;
            gt_eeg = pure_eeg;
        else
            eeg_db = pop_mergeset(eeg_db, simulated_eeg);
            gt_eeg = pop_mergeset(gt_eeg, pure_eeg);
        end

    end
    
    % Save the EEG data to the specified path
    gt_name = ['gt_db_temporal', num2str(class),'.set'];
    gt_path = ['H:\serega_db\temporal\class', num2str(class), '\'];
    pop_saveset(gt_eeg, 'filename', gt_name, 'filepath', gt_path); 

    db_name = ['sim_db_temporal', num2str(class),'.set'];
    db_path = ['H:\serega_db\temporal\class', num2str(class), '\'];
    pop_saveset(eeg_db, 'filename', db_name, 'filepath', db_path);  
    
    info_file =  sprintf('H:\\serega_db\\temporal\\class%d\\info_temporal%d.txt', class, class);
    file_id = fopen(info_file, 'w'); 

    fprintf(file_id, 'EEG synthetic data from SEREGA\n');
    fprintf(file_id, '\nLeadfield : NewYork head model (actiCAP64 projection)'); 
    fprintf(file_id, '\nTemporal class : %s\n', num2str(class)); 
    fprintf(file_id, '\nEpochs info\n'); 
    fprintf(file_id, 'N-epochs : %d x %d\n', num_data, epochs.n); 
    fprintf(file_id, 'Sampling rate : %d\n', epochs.srate);
    fprintf(file_id, 'Length of epoch : %d\n', epochs.length);
    fprintf(file_id, '\nERP info\n'); 
    fprintf(file_id, 'ERP peak latency : %d\n', erp.peakLatency); 
    fprintf(file_id, 'ERP peak latency deviation : %d\n', erp.peakLatencyDv); 
    fprintf(file_id, 'ERP peak width : %d\n', erp.peakWidth); 
    fprintf(file_id, 'ERP peak width : %d\n', erp.peakWidthDv); 
    fprintf(file_id, 'ERP peak amplitude : %d\n', erp.peakAmplitude); 
    fprintf(file_id, 'ERP peak amplitude deviation : %d\n', erp.peakAmplitudeDv);
    fprintf(file_id, '\nSource info'); 
    for i = dipole_class
        loc_source = dipole_loc(i, :);  
        source_class = lf_get_source_nearest(lf, loc_source);
        fprintf(file_id, '\nCoordinates: %.2f, %.2f, %.2f ', loc_source(1), loc_source(2), loc_source(3));
        fprintf(file_id, '(%d)', source_class );
    end 
    fprintf(file_id, '\n\nPink and white noise added with amplitude: 5 microvolt'); % and deviation 3 microvolt'); 
    fprintf(file_id, '\nSNR : %s\n', num2str(SNR)); 
    fclose(file_id);

    % Create and save sample plots
    plot_source_location(source, lf);             
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\2dsource_temporal', num2str(class), '.png']);  
    plot_source_location(source, lf, 'mode', '3d');           
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\3dsource_temporal', num2str(class), '.png']); 
    plot_source_projection(source, lf);           
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\projection_temporal', num2str(class), '.png']);  
    plot_signal_fromclass(erp, epochs); 
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\erp_temporal', num2str(class), '.png']); 
    plot_signal_fromclass(pink_noise, epochs); 
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\pink_noise_temporal', num2str(class), '.png']);
    plot_signal_fromclass(white_noise, epochs); 
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\white_noise_temporal', num2str(class), '.png']);
    pop_eegplot(pure_eeg, 1, 1, 1); 
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\ground_truth_temporal', num2str(class), '.png']);
    pop_eegplot(simulated_eeg, 1, 1, 1); 
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\db_plot_temporal', num2str(class), '.png']);
   
    disp(['All result saved at: H:\serega_db\temporal\class',num2str(class)]);
  
end

