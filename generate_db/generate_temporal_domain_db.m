% Generate simulated dataset 
% Sub: Example on temporal domain using ERP 
% Experiment design from: 
% Ravindan, A.S. & Conteras-Vidal J (2023) 
% "An Empirical Comparison of Deep Learning Explainability Approaches for EEG
% using Simulated Ground Truth"


function generate_temporal_domain_db(class, SNR) 
    % Check the number of input arguments and set default values if necessary
    if nargin < 2
        SNR = 0.4;  % Default value for SNR
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
    
    %difene 10 dipole ERP sources
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
    epochs = struct();
    epochs.n = 1000;               
    epochs.srate = 250;     
    epochs.length = 1000;   
    
    %create class in temporal domain
    erp = struct();
    if class == 1                           %positive deflection
        %random source loc
        %random_idx = randi([1, 5]);  
        random_idx = 5;
        %create erp
        erp.peakLatency = 60; 
        erp.peakLatencyDv = 8;
        erp.peakWidth = 50;  
        erp.peakWidthDv = 2;
        erp.peakAmplitude = 7; 
        erp.peakAmplitudeDv = 6;
    elseif class == 2                       %different latency
        %random_idx = randi([1, 5]);     
        random_idx = 2;
        erp.peakLatency = 900; 
        erp.peakLatencyDv = 5;
        erp.peakWidth = 100;  
        erp.peakWidthDv = 4;
        erp.peakAmplitude = 7; 
        erp.peakAmplitudeDv =6;
    elseif class == 3                       %different sign (negative)
        %random_idx = randi([1, 5]);     
        random_idx = 4;
        erp.peakLatency = 500; 
        erp.peakLatencyDv = 8;
        erp.peakWidth = 100;  
        erp.peakWidthDv = 4;
        erp.peakAmplitude = -7; 
        erp.peakAmplitudeDv =6;
    else                                    %different source position
        %random_idx = randi([6, 10]);    
        random_idx = 7;
        erp.peakLatency = 500; 
        erp.peakLatencyDv = 8;
        erp.peakWidth = 100;  
        erp.peakWidthDv = 4;
        erp.peakAmplitude = -7; 
        erp.peakAmplitudeDv =6;
    end
    
    erp = utl_check_class(erp, 'type', 'erp');

    %create pure eeg from erp
    random_loc = dipole_loc(random_idx, :);         
    source = lf_get_source_nearest(lf, random_loc);

    %create erp component 
    pure_comp = struct();
    pure_comp.source = source;
    pure_comp.signal = {erp} ;
    
    %project eeg source with erp 
    pure_scalpdata = generate_scalpdata(pure_comp, lf, epochs);

    %pink noise (represents source noise)
    pink_noise = struct( ...
            'type', 'noise', ...
            'color', 'pink', ...
            'amplitude', 5, ...
            'amplitudeDv', 3 );
    %pink_noise = utl_create_component(random_noise, pink_noise, lf);
    pink_noise = utl_check_class(pink_noise);
    
    %create projected erp-internal noise component 
    erp_comp = struct();
    erp_comp.source = source;
    erp_comp.signal = {erp, pink_noise} ;
    erp_comp = utl_check_component(erp_comp, lf);

    %project eeg source with erp 
    source_scalpdata = generate_scalpdata(erp_comp, lf, epochs);
    
    %white noise (represents sensor noise)
    white_noise = struct( ...
            'type', 'noise', ...
            'color', 'white', ...
            'amplitude', 5, ...
            'amplitudeDv', 3 );
    white_noise = utl_check_class(white_noise);
       
    %create external noise component 
    noise_comp = struct();
    noise_comp.source = source;
    noise_comp.signal = {white_noise};   
    noise_comp = utl_check_component(noise_comp, lf);
   
    %project noise to the source
    noise_scalpdata = generate_scalpdata(noise_comp, lf, epochs);      

    %add noise to signal
    EEG_snr_ratio = utl_mix_data(source_scalpdata, noise_scalpdata, snr_ratio); 
    pure_eeg =  utl_create_eeglabdataset(pure_scalpdata, epochs, lf);
    simulated_eeg = utl_create_eeglabdataset(EEG_snr_ratio, epochs, lf);
    
    % Save the EEG data to the specified path
    gt_name = ['gt_db_temp', num2str(class),'.set'];
    gt_path = ['H:\serega_db\temporal\class', num2str(class), '\'];
    pop_saveset(pure_eeg, 'filename', gt_name, 'filepath', gt_path); 

    db_name = ['sim_db_temp', num2str(class),'.set'];
    db_path = ['H:\serega_db\temporal\class', num2str(class), '\'];
    pop_saveset(simulated_eeg, 'filename', db_name, 'filepath', db_path);  
    
    info_file =  sprintf('H:\\serega_db\\temporal\\class%d\\info_temp%d.txt', class, class);
    file_id = fopen(info_file, 'w');
    
    %save info in .txt
    fprintf(file_id, 'EEG syntetic data from SEREEGA\n');
    fprintf(file_id, 'Using leadfield from NewYork head model and actiCAP64 projection'); 
    fprintf(file_id, '\nTemporal class : %s\n', num2str(class)); 
    fprintf(file_id, '\nEpochs info\n'); 
    %fprintf(file_id, 'N-epochs : %d x %d\n', num_obj, epochs.n); 
    fprintf(file_id, 'N-epochs :%d\n', epochs.n); 
    fprintf(file_id, 'Sampling rate : %d\n', epochs.srate);
    fprintf(file_id, 'Length of epoch : %d\n', epochs.length);
    fprintf(file_id, '\nSource info\n'); 
    fprintf(file_id, 'Location : %.2f\n', random_loc); 
    fprintf(file_id, 'Location idx : %d\n', source);
    fprintf(file_id, '\nERP info\n'); 
    fprintf(file_id, 'ERP peak latency : %d\n', erp.peakLatency); 
    fprintf(file_id, 'ERP peak latency deviation : %d\n', erp.peakLatencyDv); 
    fprintf(file_id, 'ERP peak width : %d\n', erp.peakWidth); 
    fprintf(file_id, 'ERP peak width : %d\n', erp.peakWidthDv); 
    fprintf(file_id, 'ERP peak amplitude : %d\n', erp.peakAmplitude); 
    fprintf(file_id, 'ERP peak amplitude deviation : %d\n', erp.peakAmplitudeDv); 
    fprintf(file_id, '\nPink and white noise added with amplitude: 5 microvolt and deviation 3 microvolt'); 
    fprintf(file_id, '\nSNR : %s\n', num2str(SNR)); 
    fclose(file_id);
     
    %if index == 1     
    % Create  and save plots
    plot_source_location(source, lf);             
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\2dsource_temp', num2str(class), '.png']);  
    plot_source_location(source, lf, 'mode', '3d');           
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\3dsource_temp', num2str(class), '.png']); 
    plot_source_projection(source, lf);           
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\projection_temp', num2str(class), '.png']);  
    plot_signal_fromclass(erp, epochs); 
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\erp_temp', num2str(class), '.png']); 
    plot_signal_fromclass(pink_noise, epochs); 
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\pink_noise_temp', num2str(class), '.png']);
    plot_signal_fromclass(white_noise, epochs); 
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\white_noise_temp', num2str(class), '.png']);
    pop_eegplot(pure_eeg, 1, 1, 1); 
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\ground_truth_temp', num2str(class), '.png']);
    pop_eegplot(simulated_eeg, 1, 1, 1); 
    saveas(gcf, ['H:\serega_db\temporal\class', num2str(class), '\db_plot_temp', num2str(class), '.png']);
    %end 
       
    %end
   
    disp(['All result saved at: H:\serega_db\temporal\class',num2str(class)]);
  
end