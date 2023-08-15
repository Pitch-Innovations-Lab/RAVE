import logging
import pdb
import math
import os
import sys

logging.basicConfig(level=logging.INFO)
logging.info("library loading")
logging.info("DEBUG")
import torch


torch.set_grad_enabled(False)

import cached_conv as cc
import gin
import nn_tilde
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from absl import flags, app
from typing import Union, Optional

try:
    import rave
except:
    import sys, os 
    sys.path.append(os.path.abspath('.'))
    import rave
import rave.blocks
import rave.core
import rave.resampler

FLAGS = flags.FLAGS

flags.DEFINE_string('run',
                    default=None,
                    help='Path to the run to export',
                    required=True)
flags.DEFINE_bool('streaming',
                  default=False,
                  help='Enable the model streaming mode')
flags.DEFINE_float(
    'fidelity',
    default=.95,
    lower_bound=.1,
    upper_bound=.999,
    help='Fidelity to use during inference (Variational mode only)')
flags.DEFINE_string('output', 
                     default= None,
                     help = "")
flags.DEFINE_bool('ema_weights',
                  default=False,
                  help='Use ema weights if avaiable')
flags.DEFINE_integer('channels',
                     default=None,
                     help = "number of out channels for export")
flags.DEFINE_integer('sr',
                     default=None,
                     help='Optional resampling sample rate')



class ScriptedRAVE(nn_tilde.Module):

    def __init__(self,
                 pretrained: rave.RAVE,
                 channels: Optional[int] = None,
                 fidelity: float = .95,
                 target_sr: bool = None) -> None:
        super().__init__()
        self.pqmf = pretrained.pqmf
        self.sr = pretrained.sr
        self.spectrogram = pretrained.spectrogram
        self.resampler = None
        self.input_mode = pretrained.input_mode
        self.output_mode = pretrained.output_mode
        self.n_channels = pretrained.n_channels
        self.target_channels = channels or self.n_channels

        if target_sr is not None:
            if target_sr != self.sr:
                assert not target_sr % self.sr, "Incompatible target sampling rate"
                self.resampler = rave.resampler.Resampler(target_sr, self.sr)
                self.sr = target_sr

        self.full_latent_size = pretrained.latent_size
        self.is_using_adain = False
        for m in self.modules():
            if isinstance(m, rave.blocks.AdaptiveInstanceNormalization):
                self.is_using_adain = True
                break
        if self.is_using_adain and (self.n_channels != self.target_channels):
            raise ValueError("AdaIN requires the original number of channels")

        self.register_attribute("learn_target", False)
        self.register_attribute("reset_target", False)
        self.register_attribute("learn_source", False)
        self.register_attribute("reset_source", False)

        self.register_buffer("latent_pca", pretrained.latent_pca)
        self.register_buffer("latent_mean", pretrained.latent_mean)
        self.register_buffer("fidelity", pretrained.fidelity)

        if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
            latent_size = max(
                np.argmax(pretrained.fidelity.numpy() > fidelity), 1)
            latent_size = 2**math.ceil(math.log2(latent_size))
            self.latent_size = latent_size

        elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
            self.latent_size = pretrained.encoder.num_quantizers

        elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
            self.latent_size = pretrained.latent_size

        elif isinstance(pretrained.encoder, rave.blocks.SphericalEncoder):
            self.latent_size = pretrained.latent_size - 1

        else:
            raise ValueError(
                f'Encoder type {pretrained.encoder.__class__.__name__} not supported'
            )

        self.fake_adain = rave.blocks.AdaptiveInstanceNormalization(0)

        # have to init cached conv before graphing
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder
        x_len = 2**14
        x = torch.zeros(1, self.n_channels, x_len)
        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)
        z = self.encode(x)
        ratio_encode = x_len // z.shape[-1]

        # configure encoder
        if pretrained.input_mode == "pqmf":
            encode_shape = (self.pqmf.n_band, 2**14 // self.pqmf.n_band) 
        else:
            encode_shape = (pretrained.n_channels, 2**14) 

        self.register_method(
            "encode",
            in_channels=self.n_channels,
            in_ratio=1,
            out_channels=self.latent_size,
            out_ratio=ratio_encode,
            input_labels=['(signal) Channel %d'%d for d in range(1, self.n_channels+1)],
            output_labels=[
                f'(signal) Latent dimension {i + 1}'
                for i in range(self.latent_size)
            ],
        )
        self.register_method(
            "decode",
            in_channels=self.latent_size,
            in_ratio=ratio_encode,
            out_channels=self.target_channels,
            out_ratio=1,
            input_labels=[
                f'(signal) Latent dimension {i+1}'
                for i in range(self.latent_size)
            ],
            output_labels=['(signal) Channel %d'%d for d in range(1, self.target_channels+1)]
        )

        self.register_method(
            "forward",
            in_channels=self.n_channels,
            in_ratio=1,
            out_channels=self.target_channels,
            out_ratio=1,
            input_labels=['(signal) Channel %d'%d for d in range(1, self.n_channels + 1)],
            output_labels=['(signal) Channel %d'%d for d in range(1, self.target_channels+1)]
        )

    def post_process_latent(self, z):
        raise NotImplementedError

    def pre_process_latent(self, z):
        raise NotImplementedError

    def update_adain(self):
        for m in self.modules():
            if isinstance(m, rave.blocks.AdaptiveInstanceNormalization):
                m.learn_x.zero_()
                m.learn_y.zero_()

                if self.learn_target[0]:
                    m.learn_y.add_(1)
                if self.learn_source[0]:
                    m.learn_x.add_(1)

                if self.reset_target[0]:
                    m.reset_y()
                if self.reset_source[0]:
                    m.reset_x()

        self.reset_source = False,
        self.reset_target = False,

    @torch.jit.export
    def encode(self, x):
        if self.is_using_adain:
            self.update_adain()

        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        batch_size = x.shape[:-2]
        if self.input_mode == "pqmf":
            x = x.reshape(-1, 1, x.shape[-1])
            x = self.pqmf(x)
            x = x.reshape(batch_size + (-1, x.shape[-1]))
        elif self.input_mode == "mel":
            x = self.spectrogram(x)[..., :-1]
            x = torch.log1p(x).reshape(batch_size + (-1, x.shape[-1]))
        z = self.encoder(x)
        z = self.post_process_latent(z)
        return z

    @torch.jit.export
    def decode(self, z, from_forward: bool = False):
        if self.is_using_adain and not from_forward:
            self.update_adain()

        if self.target_channels > self.n_channels:
            # z = torch.cat([z, z], 0)
            z = z.repeat(math.ceil(self.target_channels / self.n_channels), 1, 1)[:self.target_channels]

        z = self.pre_process_latent(z)
        y = self.decoder(z)

        batch_size = z.shape[:-2]
        if self.pqmf is not None:
            y = y.reshape(y.shape[0] * self.n_channels, -1, y.shape[-1])
            y = self.pqmf.inverse(y)
            y = y.reshape(batch_size+(self.n_channels, -1))

        if self.resampler is not None:
            y = self.resampler.from_model_sampling_rate(y)

        if self.target_channels > self.n_channels:
            y = torch.cat(y.chunk(self.target_channels, 0), 1)
        elif self.target_channels < self.n_channels:
            y = y[:, :self.target_channels]
        return y

    def forward(self, x):
        return self.decode(self.encode(x), from_forward=True)

    @torch.jit.export
    def get_learn_target(self) -> bool:
        return self.learn_target[0]

    @torch.jit.export
    def set_learn_target(self, learn_target: bool) -> int:
        self.learn_target = (learn_target, )
        return 0

    @torch.jit.export
    def get_learn_source(self) -> bool:
        return self.learn_source[0]

    @torch.jit.export
    def set_learn_source(self, learn_source: bool) -> int:
        self.learn_source = (learn_source, )
        return 0

    @torch.jit.export
    def get_reset_target(self) -> bool:
        return self.reset_target[0]

    @torch.jit.export
    def set_reset_target(self, reset_target: bool) -> int:
        self.reset_target = (reset_target, )
        return 0

    @torch.jit.export
    def get_reset_source(self) -> bool:
        return self.reset_source[0]

    @torch.jit.export
    def set_reset_source(self, reset_source: bool) -> int:
        self.reset_source = (reset_source, )
        return 0


class VariationalScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        z = self.encoder.reparametrize(z)[0]
        z = z - self.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, self.latent_pca.unsqueeze(-1))
        z = z[:, :self.latent_size]
        return z

    def pre_process_latent(self, z):
        noise = torch.randn(
            z.shape[0],
            self.full_latent_size - self.latent_size,
            z.shape[-1],
        ).type_as(z)
        z = torch.cat([z, noise], 1)
        z = F.conv1d(z, self.latent_pca.T.unsqueeze(-1))
        z = z + self.latent_mean.unsqueeze(-1)
        return z


class DiscreteScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        z = self.encoder.rvq.encode(z)
        return z.float()

    def pre_process_latent(self, z):
        z = torch.clamp(z, 0,
                        self.encoder.rvq.layers[0].codebook_size - 1).long()
        z = self.encoder.rvq.decode(z)
        if self.encoder.noise_augmentation:
            noise = torch.randn(z.shape[0], self.encoder.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)
        return z


class WasserteinScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        return z

    def pre_process_latent(self, z):
        if self.encoder.noise_augmentation:
            noise = torch.randn(z.shape[0], self.encoder.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)
        return z


class SphericalScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        return rave.blocks.unit_norm_vector_to_angles(z)

    def pre_process_latent(self, z):
        return rave.blocks.angles_to_unit_norm_vector(z)


def main(argv):
    cc.use_cached_conv(FLAGS.streaming)

    logging.info("building rave")

    gin.parse_config_file(os.path.join(FLAGS.run, "config.gin"))
    checkpoint = rave.core.search_for_run(FLAGS.run)

    pretrained = rave.RAVE()
    if checkpoint is not None:
        print('model found : %s'%checkpoint)
        checkpoint = torch.load(checkpoint, map_location='cpu')
        if FLAGS.ema_weights and "EMA" in checkpoint["callbacks"]:
            pretrained.load_state_dict(
                checkpoint["callbacks"]["EMA"],
                strict=False,
            )
        else:
            pretrained.load_state_dict(
                checkpoint["state_dict"],
                strict=False,
            )
    else:
        print("No checkpoint found")
        exit()
    pretrained.eval()

    if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
        script_class = VariationalScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
        script_class = DiscreteScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
        script_class = WasserteinScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.SphericalEncoder):
        script_class = SphericalScriptedRAVE
    else:
        raise ValueError(f"Encoder type {type(pretrained.encoder)} "
                         "not supported for export.")

    logging.info("warmup pass")

    x = torch.zeros(1, pretrained.n_channels, 2**14)
    pretrained(x)

    logging.info("optimize model")

    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)
    logging.info("script model")

    scripted_rave = script_class(
        pretrained=pretrained,
        channels = FLAGS.channels,
        fidelity=FLAGS.fidelity,
        target_sr=FLAGS.sr,
    )

    logging.info("save model")
    output = FLAGS.output or FLAGS.run
    model_name = os.path.basename(os.path.normpath(FLAGS.run))
    if FLAGS.streaming:
        model_name += "_streaming"
    model_name += ".ts"

    scripted_rave.export_to_ts(os.path.join(output, model_name))

    logging.info(
        f"all good ! model exported to {os.path.join(FLAGS.run, model_name)}")


if __name__ == "__main__":
    app.run(main)