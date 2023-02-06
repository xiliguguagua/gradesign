import numpy as np


class UserShuffler:

    def __init__(self, args):
        self.k = args.K
        self.uids = []
        self.user_num = 0
        self.triggered = False

    def add_user(self, user):
        self.uids.append(user.uid)
        self.user_num += 1
        if self.user_num >= self.k:
            self.triggered = True

    def reset(self):
        self.uids = []
        self.triggered = False


class ModelShuffler:

    def __init__(self, args):
        self.m = args.M
        self.shufflers = None
        self.grads = None
        self.ns = np.zeros(args.N)

    def collect_usershuffler(self, shufflers):
        self.shufflers = shufflers

    def split_upload(self):
        pass