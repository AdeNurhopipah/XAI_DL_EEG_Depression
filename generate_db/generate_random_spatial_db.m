% Generate simulated EEG dataset
% Example: Spatial domain using combined ERSP and ERP features
% Based on the design from: Ravindran, A.S. & Contreras-Vidal, J.L. (2023)

% Author: Ade Nurhopipah
% Email: ade.nurhopipah@postgrad.otago.ac.nz


function generate_random_spatial(class, SNR)
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
    
    %difene 14 dipole signal sources
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
    disp(snr_ratio);
    
    %create a structured epoch array
    num_data = 100;
    epochs = struct();
    epochs.n = 100;               
    epochs.srate = 250;     
    epochs.length = 1000; 

     %Create erp
    erp = struct();
    erp.type = 'erp';
        
    %create ersp
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
        
        %random spectral signal
        random_sig = randi([1, 8]); 
        disp (['Random signal:', num2str(random_sig)])

        if random_sig == 1%}
            %erp deflection
            erp.peakLatency = 60; 
            erp.peakLatencyDv = 8;
            erp.peakWidth = 50;  
            erp.peakWidthDv = 2;
            erp.peakAmplitude = 7; 
            erp.peakAmplitudeDv = 6;
        elseif random_sig == 2   
            %different latency erp
            erp.peakLatency = 900; 
            erp.peakLatencyDv = 5;
            erp.peakWidth = 100;  
            erp.peakWidthDv = 4;
            erp.peakAmplitude = 7; 
            erp.peakAmplitudeDv =6;
        elseif random_sig == 3   
            %different sign (negative)   
            erp.peakLatency = 500; 
            erp.peakLatencyDv = 8;
            erp.peakWidth = 100;  
            erp.peakWidthDv = 4;
            erp.peakAmplitude = -7; 
            erp.peakAmplitudeDv = 6;
        elseif random_sig == 4  
            %different source position   
            erp.peakLatency = 500; 
            erp.peakLatencyDv = 8;
            erp.peakWidth = 100;  
            erp.peakWidthDv = 4;
            erp.peakAmplitude = -7; 
            erp.peakAmplitudeDv =6;
        elseif random_sig == 5 
            ersp.frequency =  [2 3 8 9] ; %tetha  
        elseif random_sig == 6 
            ersp.frequency =  [7 8 13 14]; %alpha
        elseif random_sig == 7
            ersp.frequency =  [13 14 30 31]; %beta
        else
            ersp.frequency =  [49 50 58 59] ; %gamma
        end

        %pick random source
        if class == 1
            random_source = 2*randi(7) - 1; %left hemisphere
        else
            random_source = 2*randi(7);     %right hemisphere
        end

        random_loc = dipole_loc(random_source, :);         
        source = lf_get_source_nearest(lf, random_loc);
        disp(['Random location %s : %d\n', num2str(random_loc), source]);
       
        %create erp or ersp component 
        pure_comp = struct();
        pure_comp.source = source;
        
        if random_sig <5
            pure_comp.signal = {erp} ;
            erp = utl_check_class(erp, 'type', 'erp');
            disp('Generate ERP')
        else
            pure_comp.signal = {ersp} ;
            ersp = utl_check_class(ersp,'type', 'ersp');
            disp('Generate ERSP')     
        end
     
                
        %project eeg source 
        pure_scalpdata = generate_scalpdata(pure_comp, lf, epochs);
                  
        %create external noise component 
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
        
        if i == 1
            eeg_db = simulated_eeg;
            gt_eeg = pure_eeg;
        else
            eeg_db = pop_mergeset(eeg_db, simulated_eeg);
            gt_eeg = pop_mergeset(gt_eeg, pure_eeg);
        end

    end
        
    % Save the EEG data to the specified path
    gt_name = ['gt_db_spatial', num2str(class),'.set'];
    gt_path = ['H:\serega_db\spatial\class', num2str(class), '\'];
    pop_saveset(gt_eeg, 'filename', gt_name, 'filepath', gt_path); 

    db_name = ['sim_db_spatial', num2str(class),'.set'];
    db_path = ['H:\serega_db\spatial\class', num2str(class), '\'];
    pop_saveset(eeg_db, 'filename', db_name, 'filepath', db_path);  
        
    %save info in .txt
    info_file =  sprintf('H:\\serega_db\\spatial\\class%d\\info_spatial%d.txt', class, class);
    fileID = fopen(info_file, 'w');
    fprintf(fileID, 'EEG synthetic data from SEREGA\n');
    fprintf(fileID, '\nLeadfield : NewYork head model (actiCAP64 projection)'); 
    fprintf(fileID, '\nSpatial class : %s\n', num2str(class)); 
    fprintf(fileID, '\nEpochs info\n'); 
    fprintf(fileID, 'N-epochs : %d x %d\n', num_data, epochs.n);
    fprintf(fileID, 'Sampling rate : %d\n', epochs.srate);
    fprintf(fileID, 'Length of epoch : %d\n', epochs.length);
    fprintf(fileID, '\nERSP info'); 
    fprintf(fileID, '\nERSP with random properties'); 
    fprintf(fileID, '\nERP info'); 
    fprintf(fileID, '\nERP with random properties');
    fprintf(fileID, '\n\nSource info:'); 
    for i = 1:7
        if class == 1
            idx = i*2-1 ;
        else 
            idx = i*2;
        end
        loc_source = dipole_loc(idx, :);  
        source_class = lf_get_source_nearest(lf, loc_source);
        fprintf(fileID, '\nCoordinates: %.2f, %.2f, %.2f ', loc_source(1), loc_source(2), loc_source(3));
        fprintf(fileID, '(%d)', source_class );
    end 
    fprintf(fileID, '\n\nPink and white noise added with amplitude: 5 microvolt'); 
    fprintf(fileID, '\nSNR : %s\n', num2str(SNR)); 
    fclose(fileID);

    % Create and save sample plots
    plot_source_location(source, lf);             
    saveas(gcf, ['H:\serega_db\spatial\class', num2str(class), '\2dsource_spatial', num2str(class), '.png']);  
    plot_source_location(source, lf, 'mode', '3d');           
    saveas(gcf, ['H:\serega_db\spatial\class', num2str(class), '\3dsource_spatial', num2str(class), '.png']); 
    plot_source_projection(source, lf);           
    saveas(gcf, ['H:\serega_db\spatial\class', num2str(class), '\projection_spatial', num2str(class), '.png']);  
    pop_eegplot(simulated_eeg, 1, 1, 1); 
    saveas(gcf, ['H:\serega_db\spatial\class', num2str(class), '\db_plot_spatial', num2str(class), '.png']);
    pop_eegplot(pure_eeg, 1, 1, 1); 
    saveas(gcf, ['H:\serega_db\spatial\class', num2str(class), '\ground_truth_spatial', num2str(class), '.png']);
         
    disp(['All result saved at: H:\serega_db\spatial\class',num2str(class)]);
  
end

SNR = -3.5;
classes = [1,2];
for class_id = classes
    generate_random_spatial(class_id, SNR);
end
