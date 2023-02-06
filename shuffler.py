import numpy as np


class UserShuffler:

    def __init__(self, args):
        self.k = args.K
        self.uids = []
        self.grads = []
        self.user_num = 0
        self.grad_num = 0
        self.triggered = False

    def add_user(self, user):
        self.uids.append(user.id)
        self.user_num += 1
        if self.user_num >= self.k:
            self.triggered = True
        user_grad = user.clipping_perturbation()
        if user_grad is not None:  # grad is None --> user is banned, haven't trained locally
            self.grads.append(user_grad)  # so grad is zero, <add zero * lr> == <not add>
            self.grad_num += 1            # self.grad will be directly sent to beta_VAE
                                          # prevent to be detected again

    def reset(self):
        self.uids = []
        self.grads = []
        self.user_num = 0
        self.grad_num = 0
        self.triggered = False


class ModelShuffler:

    def __init__(self, args):
        self.m = args.M
        self.shufflers = None
        self.grads = []

    def collect_usershuffler(self, shufflers):
        self.shufflers = shufflers

    def split_upload(self):
        self.grads = []
        for shflr in self.shufflers:
            if not shflr.triggered:
                continue
            self.grads += shflr.grads