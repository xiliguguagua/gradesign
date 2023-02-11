import numpy as np
from utils import *

class UserShuffler:

    def __init__(self, args):
        self.k = args.k
        self.uids = set()
        self.u_weights = []
        self.user_num = 0
        self.triggered = False

    def add_user(self, user):
        self.uids.add(user.id)
        self.user_num += 1
        if self.user_num >= self.k:
            self.triggered = True
        self.u_weights.append(user.weights)

    def reset(self):
        self.uids.clear()
        self.u_weights = []
        self.user_num = 0
        self.triggered = False


class ModelShuffler:

    def __init__(self, args):
        self.m = args.M
        self.shufflers = None
        self.m_weights = []
        self.ordered_idsets = []

    def collect_usershuffler(self, shufflers):
        self.shufflers = shufflers


    def split_upload(self, server):
        self.ordered_idsets = []
        self.m_weights = []
        for shflr in self.shufflers:
            if not shflr.triggered:
                continue
            self.m_weights += shflr.u_weights
            self.ordered_idsets += shflr.uids

        self.shuffle_weights()
        self.upload(server)

    def shuffle_weights(self):
        shuffle(self.m_weights)

    def upload(self, server):
        server.rebuild_weights(self.m_weights)
