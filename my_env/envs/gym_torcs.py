import gym
from gym import spaces
import numpy as np
# from os import path
from . import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time
import pyautogui
import subprocess
import math
from collections import OrderedDict

np.set_printoptions(suppress=True, precision=4)

class TorcsEnv(gym.Env):
    terminal_judge_start = 100  # If after 100 timestep still no progress, terminated
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True

    def __init__(self, vision=False, throttle=False, gear_change=False):

        self.alpha	= 10.0#0.5
        self.beta	= -0.5

        #windows or linux
        #for windows
        if os.name == 'nt':
            self.windows = True
        #for linux
        else:
            self.windows = False

        #set environment settings
        self.throttle = False
        self.brake = False

        self.auto_throttle = False

        self.auto_gear = False

        self.vision = vision

        self.gear_change = gear_change

        self.relaunch = False
        self.initial_run = True

        #走行距離
        self.mileage = 0

        ##print("launch torcs")
        if self.windows:
            os.system('kill_torcs.bat')
            time.sleep(0.5)
            if self.vision is True:
                subprocess.Popen('torcs.bat -nofuel -nolaptime -vision &')
            else:
                subprocess.Popen('torcs.bat -nofuel -nolaptime &')

            time.sleep(0.5)
            pyautogui.press('enter', presses=2, interval=0.2)
            pyautogui.press('up', presses=2, interval=0.2)
            pyautogui.press('enter', presses=2, interval=0.2)
            time.sleep(0.5)

        else:
            os.system('pkill torcs')
            time.sleep(0.5)
            if self.vision is True:
                os.system('torcs -nofuel -nolaptime -vision &')
            else:
                os.system('torcs -nofuel -nolaptime &')
            time.sleep(0.5)
            os.system('sh autostart.sh')
            time.sleep(0.5)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        if self.throttle is False:
            high = np.array([1])
            low = np.array([-1])
        else:
            if self.brake is False:
                high = np.array([1,1])
                low = np.array([-1,0])
            else:
                high = np.array([1,1,1])
                low = np.array([-1,0,0])

        #self.action_space = spaces.Box(low=low, high=high, shape=(high.shape[0],))


        self.action_space = spaces.Box(low=low, high=high)

        #set observatiuon space
        #high = np.full(29, 1)

        high = np.full(4,1)
        low = np.full(4,-1)

        # low = np.array([-1])
        # low = np.append(low, np.full(19, 0))
        # low = np.append(low, np.full(4, -1))
        # low = np.append(low, np.full(5, 0))

        #low = np.array(np.array([-1]), np.full(19, 0), np.full(4, -1), np.full(5, 0))
        #low = np.array([[-1], np.full(19, 0), np.full(4, -1), np.full(5, 0)])

        low = low.flatten()

        self.observation_space = spaces.Box(low=low, high=high)

        print (self.observation_space.shape)

    def step(self, u):
        #print (u)
        #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]




        #action_torcs['steer'] = np.asarray([0.0])



        #走行距離を記録
        self.mileage = client.S.d['distFromStart']

        # if (self.mileage > 3000):
        #     self.mileage = 0

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            if self.auto_throttle:
                target_speed = self.default_speed
                if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                    client.R.d['accel'] += .01
                else:
                    client.R.d['accel'] -= .01

                if client.R.d['accel'] > 0.2:
                    client.R.d['accel'] = 0.2

                if client.S.d['speedX'] < 10:
                    client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

                # Traction Control System
                if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
                   (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                    action_torcs['accel'] -= .2

            #スロットルなし、速度固定の場合----------------------------------------------------------------
            else:
                action_torcs['accel'] = 0.8
        #スロットルあり
        else:
            action_torcs['accel'] = this_action['accel']
            if self.brake:
                action_torcs['brake'] = this_action['brake']

        if self.auto_gear:
            action_torcs['gear'] = 1
            if client.S.d['speedX'] > 50:
                action_torcs['gear'] = 2
            if client.S.d['speedX'] > 80:
                action_torcs['gear'] = 3
            if client.S.d['speedX'] > 110:
                action_torcs['gear'] = 4
            if client.S.d['speedX'] > 140:
                action_torcs['gear'] = 5
            if client.S.d['speedX'] > 170:
                action_torcs['gear'] = 6
        else:
            action_torcs['gear'] = 1

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d
        self.raw_data = obs

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        reward = progress

        #reward = - abs(this_action['steer'])

        #reward = self.predict(obs['angle'], obs['trackPos'])
        #reward = -abs(u[0])


        # if self.mileage > 1000:
        #     print("GOAL!")
        #     reward += 100
        #     episode_terminate = True
        #     client.R.d['meta'] = True


        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -20
            episode_terminate = True
            client.R.d['meta'] = True

        # Termination judgement #########################
        episode_terminate = False
        #if (abs(track.any()) > 1 or abs(trackPos) > 1):  # Episode is terminated if the car is out of track
        #    reward = -200
        #    episode_terminate = True
        #    client.R.d['meta'] = True

        # if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
        #    if progress < self.termination_limit_progress:
        #        print("No progress")
        #        episode_terminate = True
        #        client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        ob = self.get_obs()

        #state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        state = np.hstack((ob.angle, ob.trackPos, ob.speedX, ob.speedY))
        #state = np.hstack((ob.angle, ob.trackPos))
        #print (state)

        return state, reward, client.R.d['meta'], {}

#reward setting---------------------------------------------------------------
    def predict(self, _angle, _pos_x):

        # pos_x = _state_vector[20]
        # vel_x = _state_vector[21]
        angle = _angle
        pos_x = _pos_x
        # angle = _state_vector[0]

        # ■reward_track:中心からの距離によるreward(Risk)
        risk_trackpos = - (self.alpha) * math.fabs(pos_x * pos_x)  # + beta
        if math.fabs(pos_x) < 0.2:
            risk_trackpos = 0.0  # 中心付近はRiskなしとする

        risk_stability = - np.absolute(angle)
        # benefit_speed	= vel_x * 0.05

        r = risk_trackpos + risk_stability
        # print("reward:", r)

        return r

    def render(self, mode='human'):
        pass

    def set_reset(self):
        self.relaunch = True

    def get_raw_data(self):
        return self.raw_data

    def get_mileage(self):
        return  self.mileage

    def reset(self):
        #print("Reset")

        print (self.relaunch)

        self.mileage = 0

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if self.relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        self.relaunch = False

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3001, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False

        ob = self.get_obs()

        #状態観測いじるならここ------------------------------------------------------------------------------------------

        #state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        state = np.hstack((ob.angle, ob.trackPos, ob.speedX, ob.speedY))
        #state = np.hstack((ob.angle, ob.trackPos))

        return state

    def end(self):
        if self.windows:
            os.system('kill_torcs.bat')
        else:
            os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):

        if self.windows:
            os.system('kill_torcs.bat')
            time.sleep(0.5)
            if self.vision is True:
                # os.system('torcs.bat -nofuel -nodamage -nolaptime -vision &')
                subprocess.Popen('torcs.bat -nofuel -nolaptime -vision &')
            else:
                # os.system('torcs.bat -nofuel -nolaptime &')
                subprocess.Popen('torcs.bat -nofuel -nolaptime &')

            time.sleep(0.5)
            pyautogui.press('enter', presses=2, interval=0.2)
            pyautogui.press('up', presses=2, interval=0.2)
            pyautogui.press('enter', presses=2, interval=0.2)
            time.sleep(0.5)

        else:
            os.system('pkill torcs')
            time.sleep(0.5)
            if self.vision is True:
                os.system('torcs -nofuel -nolaptime -vision &')
            else:
                os.system('torcs -nofuel -nolaptime &')
            time.sleep(0.5)
            os.system('sh autostart.sh')
            time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})
            if self.brake:
                torcs_action.update({'brake': u[2]})

        if self.gear_change is True: # gear change action is enabled
            pass
            #torcs_action.update({'gear': int(u[3])})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)
        return np.array([r, g, b], dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel']

            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               damage=np.array(raw_obs['damage'], dtype=np.float32),
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
