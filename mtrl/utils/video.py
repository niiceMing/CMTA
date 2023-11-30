# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
"""Utility to record the environment frames into a video."""
import os
from textwrap import wrap
import cv2
import imageio
import pdb

class VideoRecorder(object):
    def __init__(self, dir_name, height=360, width=360, camera='topview', fps=30):
        """Class to record the environment frames into a video.

        Args:
            dir_name ([type]): directory to save the recording.
            height (int, optional): height of the frame. Defaults to 256.
            width (int, optional): width of the frame. Defaults to 256.
            camera_id (int, optional): id of the camera for recording. Defaults to 0.
            fps (int, optional): frames-per-second for the recording. Defaults to 30.
        """
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera = camera # one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
        self.fps = fps
        self.frames = []
        self.res = (height,width)


    def init(self, vec_env, enabled=True):
        """Initialize the recorder.

        Args:
            enabled (bool, optional): should enable the recorder or not. Defaults to True.
        """
        # self.frames = []
        self.enabled = self.dir_name is not None and enabled
        self.vec_env = []
        self.writers=[]
        self.writers=[ None for _ in range(10)]
        for env_id in range(vec_env.num_envs):
            env = vec_env.env_fns[env_id]()
            obs = env.reset()
            print(f'id:{env_id},{obs}')
            self.vec_env.append(env)
           
            self.writers[env_id] = cv2.VideoWriter(
                f'{self.dir_name}/{env_id}.avi',
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, self.res)
            # self.writers.append(writer)
            # self.writers.append(writer)
        # pdb.set_trace()    

    # def init(self, vec_env, enabled=True):
    #     """Initialize the recorder.

    #     Args:
    #         enabled (bool, optional): should enable the recorder or not. Defaults to True.
    #     """
    #     # self.frames = []
    #     self.enabled = self.dir_name is not None and enabled
    #     self.writers=[]
    #     for env_id in range(vec_env.num_envs):
    #         writer = cv2.VideoWriter(
    #             f'{self.dir_name}/{env_id}.avi',
    #             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, self.res)
    #         self.writers.append(writer)

    def sim_record(self, vec_env, action):
        """simulate and record

        Args:
            env ([type]): environment to record the frames.
              vec_env.env_fns[env_id]().env ->  single metaworld env
        """
        if self.enabled:
            for i in range(vec_env.num_envs):

                env = self.vec_env[i]
                _, _, _, _ = env.step(action[i])
                frame = env.sim.render(*self.res, mode='offscreen', camera_name=self.camera)[:,:,::-1]
                self.writers[i].write(frame)


    def relase(self):
        pdb.set_trace()
        for wirter in self.writers:
            wirter.release()


    def save(self, file_name):
        """Save the frames as video to `self.dir_name` in a file named `file_name`.

        Args:
            file_name ([type]): name of the file to store the video frames.
        """
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
