from servir.utils.download_data import get_event_for_forecasting, get_gpm_files, initialize_event_folders
from servir.utils.m_tif2h5py import tif2h5py
from servir.utils.m_h5py2tif import h5py2tif
from servir.core.model_picker import ModelPicker
import pandas as pd
from servir.utils.evaluation import generate_outputs
from servir.core.data_provider import IMERGDataModule

event_filename = '/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/forecasting_events.csv'
data_df = pd.read_csv(event_filename).dropna(how='all')

data_df['Simulation Start'] = pd.to_datetime(data_df['Simulation Start'])
data_df['Simulation End'] = pd.to_datetime(data_df['Simulation End'])

domain_location = 'WA'

# Domain coordinates for WA 
xmin = -21.4
xmax = 30.4
ymin = -2.9
ymax = 33.1

# Domain coordinates for Ghana 
# xmin = -3.95
# xmax = 2.35
# ymin = 4.85
# ymax = 11.15

        
# The outputs would assume the following shape
req_shape = (360, 518)
precipBaseFolder = '/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/'


# server path for where to download IMERG data from 
ppt_server_path = 'https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/'

# email associated with accound accessing the NASA product
# this is the authors email and is free to use by other members
email = 'aaravamudan2014@my.fit.edu'



for event_id, row in data_df.iterrows():
        
        start_time = row['Simulation Start']
        end_time = row['Simulation Start']
        

        # start_date = "June 5, 2014"
        # end_date = "June 5, 2014"


        initial_timestamp, final_timestamp = get_event_for_forecasting(
                                                start_date = start_time,
                                                end_date = end_time,
                                                start_hour = 0,
                                                start_minute = 0,
                                                end_hour = 3,
                                                end_minute = 30,
                                                )
        

        #  the following cell creates folders if not already created (assuming that the preceding directories already exist)
        precipEventFolder = initialize_event_folders(precipBaseFolder, event_id=event_id)

        
        # the following function will download the image and store it in `precipEventFolder`
        # note that it will ignore if the file already exists in the folder

        req_height, req_width = req_shape
        get_gpm_files(precipEventFolder,initial_timestamp, final_timestamp, ppt_server_path ,email, req_height, req_width)



        # once the files are downloaded in the relevant folder, we need to convert it into h5 format
        # since it is the format that the library uses for processing data

        # this is location of the metadata folder which contains all the coordinate information that
        # will be used for visualization purposes
        meta_fname = precipEventFolder + 'metadata.json'

        tif2h5py(tif_directory=precipEventFolder, h5_fname=precipBaseFolder + str(event_id)+'.h5', meta_fname=meta_fname, x1=xmin, y1=ymin, x2=xmax, y2=ymax)


        # location of the h5 file that was generated after downloading the data
        h5_dataset_location = '/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/'+str(event_id)+'.h5'

        # as of now, we do not have IR data, so we set it None
        ir_h5_dataset_location = None

        # this string is used to determine the kind of dataloader we need to use
        # for processing individual events, we reccommend the user to keep this fixed
        dataset_type = 'wa'

        data_provider =  IMERGDataModule(
                forecast_steps = 12,
                history_steps = 8,
                imerg_filename = h5_dataset_location,
                ir_filename = ir_h5_dataset_location,
                batch_size = 32,
                image_shape = req_shape,
                normalize_data=False,
                dataset = dataset_type,
                production_mode = True)

        # for now we are treating the test dataloader as the main one since we are only interested in predicting for individual events
        data_loader = data_provider.test_dataloader()


        model_type = 'lagrangian'
        model_config_location = '/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/servir_nowcasting_examples/configs/wa_imerg/lagrangian_persistence.py'
        model_save_location = '/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/servir_nowcasting_examples/'
        use_gpu = False
        produce_ensemble_outputs = False
        model_output_location = '/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/servir_nowcasting_examples/results/'
        # naive_output = generate_outputs(data_loader, model_type, model_config_location, model_save_location, use_gpu, produce_ensemble_outputs=produce_ensemble_outputs, event_name = event_id,model_output_location = model_output_location)



        model_picker = ModelPicker(model_type=model_type, 
                                model_config_location=model_config_location, 
                                model_save_location=model_save_location,
                                use_gpu=use_gpu)
        model_picker.load_data('/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/'+str(event_id)+'.h5')

        model_picker.load_model(get_ensemble=False)

        naive_output = model_picker.predict()
        # naive_output = naive_output[None, :, :, :]


        model_picker.save_output('/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/'+str(event_id)+'_outputs.h5', naive_output, num_predictions=1)


        h5py2tif(h5_fname='/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/'+str(event_id)+'_outputs.h5', 
                meta_fname='/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/'+str(event_id)+'/metadata.json', 
                tif_directory='/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/'+str(event_id)+'/',
                num_predictions=1,
                method = 'naive')

        num_nowcasting_predictions = 7
        for i in range(num_nowcasting_predictions):
                
                tif2h5py(tif_directory=precipEventFolder, h5_fname=precipBaseFolder + str(event_id)+'.h5', meta_fname=meta_fname, x1=xmin, y1=ymin, x2=xmax, y2=ymax,
                        last_only = 8)

                data_provider =  IMERGDataModule(
                        forecast_steps = 12,
                        history_steps = 8,
                        imerg_filename = h5_dataset_location,
                        ir_filename = ir_h5_dataset_location,
                        batch_size = 32,
                        image_shape = req_shape,
                        normalize_data=False,
                        dataset = dataset_type,
                        production_mode = True)

                # for now we are treating the test dataloader as the main one since we are only interested in predicting for individual events
                data_loader = data_provider.test_dataloader()
                
                model_picker = ModelPicker(model_type=model_type, 
                                model_config_location=model_config_location, 
                                model_save_location=model_save_location,
                                use_gpu=use_gpu)
                
                model_picker.load_data('/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/'+str(event_id)+'.h5')

                model_picker.load_model(get_ensemble=False)

                naive_output = model_picker.predict()

                model_picker.save_output('/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/'+str(event_id)+'_outputs.h5', naive_output, num_predictions=1)


                h5py2tif(h5_fname='/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/'+str(event_id)+'_outputs.h5', 
                        meta_fname='/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/'+str(event_id)+'/metadata.json', 
                        tif_directory='/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/'+str(event_id)+'/',
                        num_predictions=1,
                        method = 'lagrangian')
        
        break
