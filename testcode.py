from tensorflow_privacy.privacy.optimizers import dp_optimizer

optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
          l2_norm_clip=__,
          noise_multiplier=__,
          num_microbatches=__,
          learning_rate=__)