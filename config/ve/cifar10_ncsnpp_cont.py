import os
from config.base_cifar10 import BaseHFMRIConfig
from config.dynamic_io import DynamicIOConfig
from selector.config_selector import register_config

@register_config(name='hfmri_ncsnpp_cont')
class HFMRIContConfig(BaseHFMRIConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sampling.method = 'pc'
        self.sampling.predictor = 'reverse_diffusion'
        self.sampling.corrector = 'langevin'
        self.sampling.eps = 1e-5
        
        self.model.name = 'ncsnpp'
        self.model.sde_type = 'vesde'
        self.model.continuous = True
        self.model.scale_by_sigma = True
        self.model.ema_rate = 0.999
        self.model.normalization = 'GroupNorm'
        self.model.nonlinearity = 'swish'
        self.model.nf = 64
        self.model.ch_mult = (1, 2, 2, 2)
        self.model.num_res_blocks = 4
        self.model.attn_resolutions = (32,16,8)
        self.model.resamp_with_conv = True
        self.model.conditional = False
        self.model.fir = True
        self.model.fir_kernel = [1, 3, 3, 1]
        self.model.skip_rescale = True
        self.model.resblock_type = 'biggan'
        self.model.progressive = 'none'
        self.model.progressive_input = 'residual'
        self.model.progressive_combine = 'sum'
        self.model.attention_type = 'ddpm'
        self.model.init_scale = 0.0
        self.model.fourier_scale = 16
        self.model.conv_size = 3

        self.io = IOConfig()


class IOConfig:
    def __init__(self):
        self.dataset_path = "/home/muhamed/mntdrive/Graduation Project/sauron/dataset/brain_fastMRI_DICOM/fastMRI_brain_DICOM"
        self.out_ckpt_path = "./checkpoints"
        self.out_ckpt_filename_prefix = "mri_model"
        self.use_tensorboard = True
        self.tensorboard_path = "./logs"
        self.training_from_scratch = True
        self.latest_checkpoint_file_path = None
        self.latest_checkpoint_epoch = 0
        self.out_asset_suffix = os.path.join("ve", "hfmri_ncsnpp_cont")
