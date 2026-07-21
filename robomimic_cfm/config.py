"""
Config for Conditional Flow Matching algorithm.
"""

from robomimic.config.base_config import BaseConfig

class FlowMatchingConfig(BaseConfig):
    ALGO_NAME = "flow_matching"

    def train_config(self):
        """
        Setting up training parameters for Flow Matching.

        - don't need "next_obs" from hdf5 - so save on storage and compute by disabling it
        - set compatible data loading parameters
        """
        super(FlowMatchingConfig, self).train_config()

        # disable next_obs loading from hdf5
        self.train.hdf5_load_next_obs = False

        # set compatible data loading parameters
        self.train.seq_length = 16 # should match self.algo.horizon.prediction_horizon
        self.train.frame_stack = 2 # should match self.algo.horizon.observation_horizon

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.step_every_batch = True
        self.algo.optim_params.policy.learning_rate.scheduler_type = "cosine"
        self.algo.optim_params.policy.learning_rate.num_cycles = 0.5 # number of cosine cycles (used by "cosine" scheduler)
        self.algo.optim_params.policy.learning_rate.warmup_steps = 500 # number of warmup steps (used by "cosine" scheduler)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs (used by "linear" and "multistep" schedulers)
        self.algo.optim_params.policy.learning_rate.do_not_lock_keys()
        self.algo.optim_params.policy.regularization.L2 = 1e-6          # L2 regularization strength

        # horizon parameters
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.action_horizon = 8
        self.algo.horizon.prediction_horizon = 16

        # backbone: enable exactly one of unet / transformer

        # UNet parameters (default backbone)
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256,512,1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8

        # Transformer (1D DiT) parameters - set unet.enabled=False and
        # transformer.enabled=True to use this backbone instead
        self.algo.transformer.enabled = False
        self.algo.transformer.n_emb = 256                     # embedding width (must divide by n_head)
        self.algo.transformer.n_layer = 8                     # number of DiT blocks
        self.algo.transformer.n_head = 4                      # attention heads
        self.algo.transformer.mlp_ratio = 4.0                 # FFN hidden width = n_emb * mlp_ratio
        self.algo.transformer.p_drop = 0.1                    # dropout
        self.algo.transformer.diffusion_step_embed_dim = 256  # sinusoidal time embed width
        self.algo.transformer.causal = False                  # bidirectional over the action chunk
        self.algo.transformer.cross_attn = False              # True: obs as tokens the actions cross-attend to

        # EMA parameters
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75

        # Flow Matching parameters
        self.algo.fm.sigma_min = 0.0             # minimum noise scale of the probability path (0 = rectified flow)
        self.algo.fm.num_inference_steps = 10    # number of ODE integration steps at inference time
        self.algo.fm.solver = "euler"            # ODE solver: "euler" or "midpoint"
        self.algo.fm.time_embed_scale = 100.0    # scale applied to t in [0,1] before the sinusoidal timestep embedding
