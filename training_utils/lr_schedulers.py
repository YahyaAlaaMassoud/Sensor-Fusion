
import abc
import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util.tf_export import keras_export

class LearningRateSchedule(object):
  """A serializable learning rate decay schedule.
  `LearningRateSchedule`s can be passed in as the learning rate of optimizers in
  `tf.keras.optimizers`. They can be serialized and deserialized using
  `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.
  """

  @abc.abstractmethod
  def __call__(self, step):
    raise NotImplementedError("Learning rate schedule must override __call__")

  @abc.abstractmethod
  def get_config(self):
    raise NotImplementedError("Learning rate schedule must override get_config")

  @classmethod
  def from_config(cls, config):
    """Instantiates a `LearningRateSchedule` from its config.
    Args:
        config: Output of `get_config()`.
    Returns:
        A `LearningRateSchedule` instance.
    """
    return cls(**config)

class CosineDecayRestarts(LearningRateSchedule):
  """A LearningRateSchedule that uses a cosine decay schedule with restarts."""

  def __init__(
      self,
      initial_learning_rate,
      first_decay_steps,
      t_mul=2.0,
      m_mul=1.0,
      alpha=0.0,
      name=None):
    """Applies cosine decay with restarts to the learning rate.
    See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
    with Warm Restarts. https://arxiv.org/abs/1608.03983
    When training a model, it is often recommended to lower the learning rate as
    the training progresses. This schedule applies a cosine decay function with
    restarts to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.
    The schedule a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    The learning rate multiplier first decays
    from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
    restart is performed. Each new warm restart runs for `t_mul` times more
    steps and with `m_mul` times smaller initial learning rate.
    Example usage:
    ```python
    first_decay_steps = 1000
    lr_decayed_fn = (
      tf.keras.experimental.CosineDecayRestarts(
          initial_learning_rate,
          first_decay_steps))
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
        number. The initial learning rate.
      first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python
        number. Number of steps to decay over.
      t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the number of iterations in the i-th period
      m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the initial learning rate of the i-th period:
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of the initial_learning_rate.
      name: String. Optional name of the operation.  Defaults to 'SGDRDecay'.
    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """
    super(CosineDecayRestarts, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.first_decay_steps = first_decay_steps
    self._t_mul = t_mul
    self._m_mul = m_mul
    self.alpha = alpha
    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "SGDRDecay") as name:
      initial_learning_rate = ops.convert_to_tensor(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      first_decay_steps = math_ops.cast(self.first_decay_steps, dtype)
      alpha = math_ops.cast(self.alpha, dtype)
      t_mul = math_ops.cast(self._t_mul, dtype)
      m_mul = math_ops.cast(self._m_mul, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      completed_fraction = global_step_recomp / first_decay_steps

      def compute_step(completed_fraction, geometric=False):
        """Helper for `cond` operation."""
        if geometric:
          i_restart = math_ops.floor(
              math_ops.log(1.0 - completed_fraction * (1.0 - t_mul)) /
              math_ops.log(t_mul))

          sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
          completed_fraction = (completed_fraction - sum_r) / t_mul**i_restart

        else:
          i_restart = math_ops.floor(completed_fraction)
          completed_fraction -= i_restart

        return i_restart, completed_fraction

      i_restart, completed_fraction = control_flow_ops.cond(
          math_ops.equal(t_mul, 1.0),
          lambda: compute_step(completed_fraction, geometric=False),
          lambda: compute_step(completed_fraction, geometric=True))

      m_fac = m_mul**i_restart
      cosine_decayed = 0.5 * m_fac * (1.0 + math_ops.cos(
          constant_op.constant(math.pi) * completed_fraction))
      decayed = (1 - alpha) * cosine_decayed + alpha

      return math_ops.multiply(initial_learning_rate, decayed, name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "first_decay_steps": self.first_decay_steps,
        "t_mul": self._t_mul,
        "m_mul": self._m_mul,
        "alpha": self.alpha,
        "name": self.name
    }

class CosineDecay(LearningRateSchedule):
  """A LearningRateSchedule that uses a cosine decay schedule."""

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      alpha=0.0,
      name=None):
    """Applies cosine decay to the learning rate.
    See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
    with Warm Restarts. https://arxiv.org/abs/1608.03983
    When training a model, it is often recommended to lower the learning rate as
    the training progresses. This schedule applies a cosine decay function
    to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.
    The schedule a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    It is computed as:
    ```python
    def decayed_learning_rate(step):
      step = min(step, decay_steps)
      cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
      decayed = (1 - alpha) * cosine_decay + alpha
      return initial_learning_rate * decayed
    ```
    Example usage:
    ```python
    decay_steps = 1000
    lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate, decay_steps)
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a
        Python number. The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of initial_learning_rate.
      name: String. Optional name of the operation.  Defaults to 'CosineDecay'.
    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """
    super(CosineDecay, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.alpha = alpha
    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "CosineDecay"):
      initial_learning_rate = ops.convert_to_tensor(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      decay_steps = math_ops.cast(self.decay_steps, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
      completed_fraction = global_step_recomp / decay_steps
      cosine_decayed = 0.5 * (1.0 + math_ops.cos(
          constant_op.constant(math.pi) * completed_fraction))

      decayed = (1 - self.alpha) * cosine_decayed + self.alpha
      return math_ops.multiply(initial_learning_rate, decayed)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "alpha": self.alpha,
        "name": self.name
    }

def const_lr(lr=0.0001):

    init_lr = lr
    def get_lr(epoch):
        return init_lr
    
    return get_lr


class NoisyLinearCosineDecay(LearningRateSchedule):
  """A LearningRateSchedule that uses a noisy linear cosine decay schedule."""

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      initial_variance=1.0,
      variance_decay=0.55,
      num_periods=0.5,
      alpha=0.0,
      beta=0.001,
      name=None):
    """Applies noisy linear cosine decay to the learning rate.
    See [Bello et al., ICML2017] Neural Optimizer Search with RL.
    https://arxiv.org/abs/1709.07417
    For the idea of warm starts here controlled by `num_periods`,
    see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
    with Warm Restarts. https://arxiv.org/abs/1608.03983
    Note that linear cosine decay is more aggressive than cosine decay and
    larger initial learning rates can typically be used.
    When training a model, it is often recommended to lower the learning rate as
    the training progresses. This schedule applies a noisy linear cosine decay
    function to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.
    The schedule a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    It is computed as:
    ```python
    def decayed_learning_rate(step):
      step = min(step, decay_steps)
      linear_decay = (decay_steps - step) / decay_steps)
      cosine_decay = 0.5 * (
          1 + cos(pi * 2 * num_periods * step / decay_steps))
      decayed = (alpha + linear_decay + eps_t) * cosine_decay + beta
      return initial_learning_rate * decayed
    ```
    where eps_t is 0-centered gaussian noise with variance
    initial_variance / (1 + global_step) ** variance_decay
    Example usage:
    ```python
    decay_steps = 1000
    lr_decayed_fn = (
      tf.keras.experimental.NoisyLinearCosineDecay(
        initial_learning_rate, decay_steps))
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
        number. The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      initial_variance: initial variance for the noise. See computation above.
      variance_decay: decay for the noise's variance. See computation above.
      num_periods: Number of periods in the cosine part of the decay.
        See computation above.
      alpha: See computation above.
      beta: See computation above.
      name: String.  Optional name of the operation.  Defaults to
        'NoisyLinearCosineDecay'.
    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """
    super(NoisyLinearCosineDecay, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.initial_variance = initial_variance
    self.variance_decay = variance_decay
    self.num_periods = num_periods
    self.alpha = alpha
    self.beta = beta
    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "NoisyLinearCosineDecay") as name:
      initial_learning_rate = ops.convert_to_tensor(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      decay_steps = math_ops.cast(self.decay_steps, dtype)
      initial_variance = math_ops.cast(self.initial_variance, dtype)
      variance_decay = math_ops.cast(self.variance_decay, dtype)
      num_periods = math_ops.cast(self.num_periods, dtype)
      alpha = math_ops.cast(self.alpha, dtype)
      beta = math_ops.cast(self.beta, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
      linear_decayed = (decay_steps - global_step_recomp) / decay_steps
      variance = initial_variance / (
          math_ops.pow(1.0 + global_step_recomp, variance_decay))
      std = math_ops.sqrt(variance)
      noisy_linear_decayed = (
          linear_decayed + random_ops.random_normal(
              linear_decayed.shape, stddev=std))

      completed_fraction = global_step_recomp / decay_steps
      fraction = 2.0 * num_periods * completed_fraction
      cosine_decayed = 0.5 * (
          1.0 + math_ops.cos(constant_op.constant(math.pi) * fraction))
      noisy_linear_cosine_decayed = (
          (alpha + noisy_linear_decayed) * cosine_decayed + beta)

      return math_ops.multiply(
          initial_learning_rate, noisy_linear_cosine_decayed, name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "initial_variance": self.initial_variance,
        "variance_decay": self.variance_decay,
        "num_periods": self.num_periods,
        "alpha": self.alpha,
        "beta": self.beta,
        "name": self.name
    }