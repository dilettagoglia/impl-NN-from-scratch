import math

class LearningRate:

    @staticmethod
    def linear_decay(curr_lr, init_lr, fixed_lr, curr_step, tau_step):
        """
        Function that decays linearly the learning rate eta (init_lr) for each step until iteration tau_step
        Then stops decaying and uses fix learning rate (fixed_lr)

        Args:
            curr_lr (float): learning rate from previous iteration
            init_lr (float): initial learning rate
            fixed_lr (float): final fixed learning rate
            curr_step (int): current iteration
            tau_step (int): iteration number when we stop decaying

        Returns:
            float: learning rate updated
        """
        if curr_step < tau_step and curr_lr > fixed_lr:
            # decay rate for each iteration
            alpha = curr_step / tau_step
            curr_lr = (1. - alpha) * init_lr + alpha * fixed_lr
            return curr_lr
        return fixed_lr

    @staticmethod
    def exponential_decay(init_lr, decay_rate, curr_step, decay_steps):
        """
        Function that decays exponentially the learning rate eta by 'decay_rate' every 'decay_steps', starting from 'init_lr'.
        So, we decay the learning rate by a portion of 'decay_rate' at each iteration (step).

        Args:
            init_lr (float): initial learning rate
            decay_rate (float): amount of decay at each stage
            curr_step (int): current iteration (step)
            decay_steps (int): length of each stage, composed of multiple iterations (steps)

        Returns:
            float: learning rate updated
        """ 
        alpha = curr_step / decay_steps
        return init_lr * math.exp(-decay_rate * alpha)

    @staticmethod
    def init_decay_technique(name):
        if name == "linear_decay":
            return LearningRate.linear_decay
        elif name == "exponential_decay":
            return LearningRate.exponential_decay
