"""
Test script for the flow matching algorithm. Each test trains a variant of
flow matching for a handful of gradient steps and tries one rollout with
the model. Excludes stdout output by default (pass --verbose
to see stdout output).
"""
import argparse
import json
import os
from collections import OrderedDict

import robomimic
from robomimic.config import Config
import robomimic.utils.test_utils as TestUtils
from robomimic.utils.log_utils import silence_stdout
from robomimic.utils.torch_utils import dummy_context_mgr

# registers the "flow_matching" algo + config with robomimic's factories
import robomimic_cfm  # noqa: F401

# our config template ships with this package (robomimic itself does not know
# about flow_matching, so we can't use TestUtils.get_base_config which reads the
# template from robomimic's own package dir)
_TEMPLATE_PATH = os.path.join(
    os.path.dirname(robomimic_cfm.__file__), "exps", "templates", "flow_matching.json"
)


def _test_base_config():
    """
    Mirror of robomimic's TestUtils.get_base_config, but loads this package's
    flow_matching template instead of one shipped inside robomimic.
    """
    with open(_TEMPLATE_PATH, "r") as f:
        config = Config(json.load(f))

    # small dataset with a handful of trajectories + temp model dir
    config.train.data = TestUtils.example_dataset_path()
    model_dir = TestUtils.temp_model_dir_path()
    TestUtils.maybe_remove_dir(model_dir)
    config.train.output_dir = model_dir

    # train and validate for a handful of gradient steps
    config.experiment.name = "test"
    config.experiment.validate = True
    config.experiment.epoch_every_n_steps = 3
    config.experiment.validation_epoch_every_n_steps = 3
    config.train.num_epochs = 1
    config.train.hdf5_filter_key = "train"
    config.train.hdf5_validation_filter_key = "valid"

    # exercise saving + rollout too
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = 1
    config.experiment.rollout.enabled = True
    config.experiment.rollout.rate = 1
    config.experiment.rollout.n = 1
    config.experiment.rollout.horizon = 10
    config.experiment.logging.terminal_output_to_txt = False
    config.train.cuda = True

    return config


def get_algo_base_config(video=False):
    """
    Base config for testing flow matching algorithms.
    """

    # config with basic settings for quick training run
    config = _test_base_config()

    # rollout video rendering needs a working offscreen renderer (EGL/OSMesa),
    # which headless login nodes may not have; low_dim rollouts themselves don't
    config.experiment.render_video = video

    # low-level obs (note that we define it here because @observation structure might vary per algorithm,
    # for example HBC)
    config.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    config.observation.modalities.obs.rgb = []

    # flow matching requires actions normalized to [-1, 1]
    config.train.action_config["actions"]["normalization"] = "min_max"

    return config


# mapping from test name to config modifier functions
MODIFIERS = OrderedDict()
def register_mod(test_name):
    def decorator(config_modifier):
        MODIFIERS[test_name] = config_modifier
    return decorator


@register_mod("flow_matching-euler")
def fm_euler_modifier(config):
    # no-op: rectified flow (sigma_min=0) with Euler solver is the default
    return config


@register_mod("flow_matching-midpoint")
def fm_midpoint_modifier(config):
    config.algo.fm.solver = "midpoint"
    return config


@register_mod("flow_matching-ot-cfm")
def fm_ot_cfm_modifier(config):
    # OT-CFM path with residual noise scale at t=1
    config.algo.fm.sigma_min = 0.05
    return config


@register_mod("flow_matching-no-ema")
def fm_no_ema_modifier(config):
    config.algo.ema.enabled = False
    return config


@register_mod("flow_matching-single-step")
def fm_single_step_modifier(config):
    # 1-step Euler inference (degenerate but must run)
    config.algo.fm.num_inference_steps = 1
    return config


@register_mod("flow_matching-transformer")
def fm_transformer_modifier(config):
    # 1D DiT backbone instead of the conv UNet (small dims to keep the test fast)
    config.algo.unet.enabled = False
    config.algo.transformer.enabled = True
    config.algo.transformer.n_emb = 64
    config.algo.transformer.n_layer = 2
    config.algo.transformer.n_head = 2
    return config


def test_flow_matching(silence=True, video=False):
    for test_name in MODIFIERS:
        context = silence_stdout() if silence else dummy_context_mgr()
        with context:
            base_config = get_algo_base_config(video=video)
            res_str = TestUtils.test_run(base_config=base_config, config_modifier=MODIFIERS[test_name])
        print("{}: {}".format(test_name, res_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="don't suppress stdout during tests",
    )
    parser.add_argument(
        "--video",
        action='store_true',
        help="also test rollout video rendering (requires EGL or OSMesa)",
    )
    args = parser.parse_args()

    test_flow_matching(silence=(not args.verbose), video=args.video)
