from servir.utils.evaluation import generate_outputs
from servir.core.data_provider import IMERGDataModule

# event id for which the data was downloaded
event_id = 'WA'

# location of the h5 file that was generated after downloading the data
h5_dataset_location = '/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/'+str(event_id)+'.h5'

# as of now, we do not have IR data, so we set it None
ir_h5_dataset_location = '/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/'+str(event_id)+'_IR.h5'

# this string is used to determine the kind of dataloader we need to use
# for processing individual events, we reccommend the user to keep this fixed
dataset_type = 'wa_ir'


data_provider =  IMERGDataModule(
        forecast_steps = 12,
        history_steps = 12,
        imerg_filename = h5_dataset_location,
        ir_filename = ir_h5_dataset_location,
        batch_size = 32,
        image_shape = (360, 516),
        normalize_data=False,
        dataset = dataset_type,
        production_mode = False,
        )

test_data_loader = data_provider.test_dataloader()
train_data_loader = data_provider.train_dataloader()
val_data_loader = data_provider.val_dataloader()



