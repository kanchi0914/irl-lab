#!/usr/bin/python
# -*- coding: utf-8 -*-
# snakeoil_gym.py
# Chris X Edwards <snakeoil@xed.ch>
# Snake Oil is a Python library for interfacing with a TORCS
# race car simulator which has been patched with the server
# extentions used in the Simulated Car Racing competitions.
# http://scr.geccocompetitions.com/
#
# To use it, you must import it and create a "drive()" function.
# This will take care of option handling and server connecting, etc.
# To see how to write your own client do something like this which is
# a complete working client:
# /-----------------------------------------------\
# |#!/usr/bin/python                              |
# |import snakeoil                                |
# |if __name__ == "__main__":                     |
# |    C= snakeoil.Client()                       |
# |    for step in xrange(C.maxSteps,0,-1):       |
# |        C.get_servers_input()                  |
# |        snakeoil.drive_example(C)              |
# |        C.respond_to_server()                  |
# |    C.shutdown()                               |
# \-----------------------------------------------/
# This should then be a full featured client. The next step is to
# replace 'snakeoil.drive_example()' with your own. There is a
# dictionary which holds various option values (see `default_options`
# variable for all the details) but you probably only need a few
# things from it. Mainly the `trackname` and `stage` are important
# when developing a strategic bot. 
#
# This dictionary also contains a ServerState object
# (key=S) and a DriverAction object (key=R for response). This allows
# you to get at all the information sent by the server and to easily
# formulate your reply. These objects contain a member dictionary "d"
# (for data dictionary) which contain key value pairs based on the
# server's syntax. Therefore, you can read the following:
#    angle, curLapTime, damage, distFromStart, distRaced, focus,
#    fuel, gear, lastLapTime, opponents, racePos, rpm,
#    speedX, speedY, speedZ, track, trackPos, wheelSpinVel, z
# The syntax specifically would be something like:
#    X= o[S.d['tracPos']]
# And you can set the following:
#    accel, brake, clutch, gear, steer, focus, meta 
# The syntax is:  
#     o[R.d['steer']]= X
# Note that it is 'steer' and not 'steering' as described in the manual!
# All values should be sensible for their type, including lists being lists.
# See the SCR manual or http://xed.ch/help/torcs.html for details.
#
# If you just run the snakeoil_gym.py base library itself it will implement a
# serviceable client with a demonstration drive function that is
# sufficient for getting around most tracks.
# Try `snakeoil_gym.py --help` to get started.

import socket 
import time
import argparse
import sys
import json
import math

import numpy as np
import matplotlib.pyplot as plt
import copy
import gym
import os
import subprocess
import pyautogui
import numpy

#import common

PI= 3.14159265359

def LoadBenefitFile(_readFile, _data):
    count = 0
    for data in _readFile:
        #print datas
        data = data.strip()	#delete "\n"
        data = data.split()
         
        reward = data[1]
        #print "data[", count, "] = ", reward
        _data.append(reward)
        count += 1

class TorcsEnv4(gym.Env):

    def __init__(self):
        if os.name == 'nt':
            self.windows = True
        else:
            self.windows = False

        #set environment settings
        self.throttle = False
        self.brake = False
        self.auto_gear = False
        self.auto_throttle = False

        self.relaunch = False
        self.initial_run = True

        self.C = Client(d=True)

        self.R = Reward_Base()

        subprocess.Popen('pkill torcs', shell=True)

        if self.windows:
            os.system('kill_torcs.bat')
            time.sleep(0.5)
            subprocess.Popen("torcs -t 1000000")

            time.sleep(0.5)
            pyautogui.press('enter', presses=2, interval=0.2)
            pyautogui.press('up', presses=2, interval=0.2)
            pyautogui.press('enter', presses=2, interval=0.2)
            time.sleep(0.5)

        else:
            cmd = "torcs -t 1000000"
            proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
            time.sleep(0.20)
            os.system('sh autostart.sh')
            self.C.setup_connection()

        print ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

        #setting action space------------------------------------------
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

        self.action_space = gym.spaces.Box(low=low, high=high)

        #setting observation space---------------------------------------
        # low = np.array([-1])
        # low = np.append(low, np.full(19, 0))
        # low = np.append(low, np.full(4, -1))
        # low = np.append(low, np.full(5, 0))

        high = np.full(2,1)
        low = np.full(2,-1)

        self.observation_space = gym.spaces.Box(low=low, high=high)



        print(self.observation_space.shape, self.action_space)

    def step(self, action):
        self.C.Action['steer'] = action[0]
        act = self.C.Action
        a_vector = self.action_to_vector(act)

        self.C.update(act)

        s,a = self.C.run()
          # s, a -> s'

        #print (act)

        s_vector = self.state_to_vector(s)
        a_vector = self.action_to_vector(act)

        if a["meta"] == 1:
            r = -20
        else:
            r = self.R.predict(s_vector, a_vector)

        return s_vector, r, a["meta"], {}

    def reset(self):

        if self.C.initFlag is False:
            print("reset!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.C.Action['meta'] = 1
            self.C.restart = True
            self.C.set_preDist()

        self.C.set_preDist()

        s,a = self.C.run()
        s_vector = self.state_to_vector(s)
        #print (self.C.Action)
        return s_vector

    def action_to_vector(self, _action):
        al = []
        al.append(_action['steer'])
        #al.append(_action[0])

        return numpy.array(al, dtype=numpy.float32)

    def render(self):
        pass

    def state_to_vector(self, _state):  # ステータスはAngleとtrackPosのみ
        sl = []
        tempsl = []

        self.state_dim = self.observation_space.shape[0]

        if self.state_dim == 2:
            # [0]		1 dim
            sl.append(_state['angle'])
            sl.append(_state['trackPos'])


        elif self.state_dim == 25:
            # ALL	1+19+1+3 = 24dim
            # [0]		1 dim
            sl.append(_state['angle'])
            # sl.append(state['gear'])
            # sl.extend(state['opponents'])
            # sl.append(state['rpm'])
            # [1] - [19]	19dim
            sl.extend(_state['track'])
            # [20]		1 dim
            sl.append(_state['trackPos'])
            # [21] - [23]	3 dim
            sl.append(_state['speedX'])
            sl.append(_state['speedY'])
            sl.append(_state['speedZ'])
            # [24]		1 dim
            sl.append(_state['distFromStart'])
            # track情報正規化
            # for i in range(1, STATE_DIM - 1):
            for i in range(1, 20):
                sl[i] = sl[i] / 200.0

            # track情報の入力キャンセル
            # for i in range(1, 20):
            #    sl[i] = 0.0

            # speed情報正規化
            # for i in range(1, STATE_DIM - 1):
            for i in range(21, 24):
                sl[i] = sl[i] / 300.0  # Max 300.0km/h想定

            # speed情報の入力キャンセル
            # for i in range(21, 24):
            #    sl[i] = 0.0

            sl[24] = sl[24] / 1000.0  # 5000.0	#Max 5000想定
            sl[24] = 0.0

        # else:
        #    print("state dim error")
        #    exit(-1)

        # print("sl", sl[0], sl[20])
        return numpy.array(sl, dtype=numpy.float32)


class Reward_Base(object):

    def __init__(self):
        self.alpha = 10.0  # 0.5
        self.beta = -0.5

    # def predict(self, _pos_x, _vel_x):
    def predict(self, _state_vector, _action_vector):
        # pos_x = _state_vector[20]
        # vel_x = _state_vector[21]
        angle = _state_vector[0]
        pos_x = _state_vector[1]
        # angle = _state_vector[0]

        # ■reward_track:中心からの距離によるreward(Risk)
        risk_trackpos = - (self.alpha) * math.fabs(pos_x * pos_x)  # + beta
        if math.fabs(pos_x) < 0.2:
            risk_trackpos = 0.0  # 中心付近はRiskなしとする

        risk_stability = - numpy.absolute(angle)
        # benefit_speed	= vel_x * 0.05

        r = risk_trackpos + risk_stability
        # print("reward:", r)

        return r


class Client(object):
    def __init__(self, H=None, p=None, sid=None, e=None, t=None, s=None, d=None, _maxSteps = 1000):

        # If you don't like the option defaults,  change them here.
        self.host = H if H else 'localhost'
        self.port = p if p else 3001
        self.sid = sid if sid else 'SCR'
        self.maxEpisodes = e if e else 0
        self.trackname = t if t else 'unknown'
        self.stage = s if s else 3
        self.debug = d if d else False
        #self.maxSteps = _maxSteps#1000000000  # 50steps/second
        
        self.State = ServerState()
        
        self.Action = DriverAction()
        self.Action["meta"] = 0
        
        self.preState	= copy.copy(self.State)
        #print("self.State", self.State)
        self.preDistRaced = 0
        
        self.stuckcount	= 0
        self.deltadist	= 0
        #self.iteratorcount = 0
        
        self.restart	= False
        self.initFlag	= True
        
        #self.so
        #self.get_servers_input()

    def set_preDist(self):
        self.preDistRaced = 0

    def setup_connection(self):
        # == Set Up UDP Socket ==
        count = 0

        try:
            self.so= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error:
            print('Error: Could not create socket...', file=sys.stderr)
            sys.exit(-1)
        # == Initialize Connection To Server ==
        self.so.settimeout(1)
        while True:
            #a = "-90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90"
            a = "-90 -80 -70 -60 -50 -40 -30 -20 -10 0 10 20 30 40 50 60 70 80 90"
        
            initmsg = '%s(init %s)' % (self.sid,a)
            try:
                self.so.sendto(initmsg.encode('ascii'), (self.host, self.port))
            except socket.error:
                sys.exit(-1)
            sockdata= str()
            try:
                sockdata, addr = self.so.recvfrom(1024)
            except socket.error:
                print("Waiting for server............", file=sys.stderr)
                count += 1
                if count >= 5:
                    #self.so.close()
                    #self.so = None

                    os.system('pkill torcs')
                    time.sleep(0.5)
                    cmd = "torcs -t 1000000"
                    proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
                    time.sleep(0.20)
                    os.system('sh autostart.sh')
                    print ("relaunched")

                    break

            if '***identified***' in str(sockdata):
                print("Client connected..............", file=sys.stderr)
                break

    def get_servers_input(self): #戻り値を追加 True:再接続が必要、False:再接続不要
        '''Server's input is stored in a ServerState object'''
        self.i2 = 0
        if not self.so: return
        sockdata= str()
        while True:
            try:
                # Receive server data 
                sockdata, addr = self.so.recvfrom(1024)
                sockdata = sockdata.decode('ascii')
            except socket.error:
                print("Waiting for data..............",  file=sys.stderr)
            if '***identified***' in sockdata:
                print("Client connected..............",  file=sys.stderr)
                continue
            elif '***shutdown***' in sockdata:
                print("Server has stopped the race. You were in %d place." % self.State['racePos'],  file=sys.stderr)
                self.shutdown()
                return False
            elif '***restart***' in sockdata:
                # What do I do here?
                print("Server has restarted the race.",  file=sys.stderr)
                # I haven't actually caught the server doing this.
                #self.shutdown()
                self.Action["meta"] = 0
                return True
            elif not sockdata: # Empty?
                if self.Action["meta"] == 1:
                    return True
                continue       # Try again.
            else:
                self.State.parse_server_str(sockdata)
#                if self.debug: 
#                    print(json.dumps(self.State))
#                    print(self.State['remainlaps'])
                break # Can now return from this function.

        return False

    def respond_to_server(self):
        if not self.so: 
            return
#        if self.debug: 
#            print(json.dumps(self.Action))
#            print(self.Action)
        try:
            self.so.sendto(repr(self.Action).encode('ascii'), (self.host, self.port)) 
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1],str(emsg[0])))
            sys.exit(-1)
            
            
    #def run(self, _learn_flag = 0, _dump_flag = 0, _dump_file_name = ""):
    def run(self):

        #print (self.preDistRaced)
        #serverから情報を取得
        #print("get_servers_input")
        #print("pre", self.State['speedX'])
        need_reconnect = self.get_servers_input()

        #print ("after:",self.preDistRaced)
        
        if self.initFlag == True:
            self.preState	= copy.copy(self.State)
            self.initFlag = False
            
        #print("cur", self.State['speedX'])
        #print("run self.Action", self.Action)
        #check stuck
        if self.restart is False:
            if self.restart == True:
                self.set_preDist()
            #rint (self.preDistRaced)
            self.restart = self.checkstuck(self.State)
        #print("self.restart", self.restart)

        #リスタート処理
        if self.restart:
            self.Action["meta"] = 1
            print("restart---------------------------")
            self.restart = False
            self.preDistRaced = 0.0

        if need_reconnect:
            #print("need_reconnect---------------------------")
            self.so.close()
            print ("reconnect!!")
            self.Action["meta"] = 0
            self.setup_connection()    

        #self.iteratorcount += 1

        return self.State, self.Action

    def update(self, _action):
        self.Action = _action
        #print("respond_to_server")
        self.preState = copy.copy(self.State)

        self.respond_to_server()


    def shutdown(self):
        self.preDistRaced = 0
        if not self.so:
            return
        print("Race terminated. Shutting down.")
        self.so.close()
        self.so = None
        #sys.exit() # No need for this really.

    def checkstuck(self, status):

        #print ("befor 2", self.preDistRaced)

        ret = False
        
        #print("checkstuck")
        #course out
        pos_x = status['trackPos']
        if math.fabs(pos_x) > 1.0:
            print("course out!! -> restart")
            ret = True
        
        #reverse
        curDist = float(status['distRaced'])

        #print ("just befotr:", self.preDistRaced)

        self.deltadist = curDist - self.preDistRaced

        if np.cos(status['angle']) < 0:
            print ("turned!!")
            ret = True

        if self.deltadist < -0.2:
            print("deltadist", '%0.3f' % self.deltadist, '%0.3f' % curDist, '%0.3f' % self.preDistRaced)
            print("reverse!!")
            ret = True

        #self.preDistRaced = curDist

        #speed min
        vel_x			= status['speedX']
        dist_from_start	= status['distRaced']
        if dist_from_start > 1.0 and vel_x < 3.0:
            self.stuckcount += 1
            print("self.stuckcount", self.stuckcount)
            if self.stuckcount > 30:
                print("stuck!!")
                self.stuckcount = 0
                ret = True


        return ret

class ServerState(dict):
    'What the server is reporting right now.'
    def __init__(self):
        self.servstr= str()

    def parse_server_str(self, server_string):
        'parse the server string'
        self.servstr= server_string.strip()[:-1]
        sslisted= self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w= i.split(' ')
            self[w[0]] = destringify(w[1:])

    def __repr__(self):
        out= str()
        for k in sorted(self):
            strout= str(self[k])
            if type(self[k]) is list:
                strlist= [str(i) for i in self[k]]
                strout= ', '.join(strlist)
            out+= "%s: %s\n" % (k,strout)
        return out

class DriverAction(dict):
    '''What the driver is intending to do (i.e. send to the server).
    Composes something like this for the server:
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus 0)(meta 0) or
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus -90 -45 0 45 90)(meta 0)'''
    def __init__(self):
       self.actionstr= str()
       for k, v in (('accel', 0.8),
                    ('brake', 0),
                    ('clutch', 0),
                    ('gear', 1),
                    ('steer', 0),
                    ('focus', [-90,-45,0,45,90]),
                    ('meta', 0),
                    ):
           self[k] = v

    def __repr__(self):
        out = str()
        for k in self:
            out += '('+k+' '
            v = self[k]
            if not type(v) == list:
                out += '%.3f' % v
            else:
                out += ' '.join([str(x) for x in v])
            out += ')'
        return out
        return out+'\n'

# == Misc Utility Functions
def destringify(s):
    '''makes a string into a value or a list of strings into a list of
    values (if possible)'''
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            print("Could not find a value in %s" % s)
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]


def argument_proc():
    parser = argparse.ArgumentParser()    
    
    #"""
    parser.add_argument('-H', '--host', help='TORCS server host. [localhost]', default='localhost')
    parser.add_argument('-p', '--port', help='TORCS port. [3001]', default=3001, type=int)
    parser.add_argument('-i', '--id',  help='ID for server. [SCR]', default='SCR')
    parser.add_argument('-m', '--steps', help='Maximum simulation steps. 1 sec ~ 50 steps. [100000]', default=10000, type=int)
    parser.add_argument('-e', '--episodes', help='Maximum learning episodes. [1]', default=5, type=int)
    
    parser.add_argument('-t', '--track', help='Your name for this track. Used for learning. [unknown]', default='unknown')
    parser.add_argument('-s', '--stage', choices=(0, 1, 2, 3), help='0=warm up, 1=qualifying, 2=race, 3=unknown. [3]', default=3, type=int)
    parser.add_argument('-d', '--debug', help='Output full telemetry.', action='store_true')
    
    parser.add_argument('-l', '--learn', help='Learning Agent.', action='store_true')
    #"""
    
    print("parser.parse_args()", parser.parse_args())
    return parser.parse_args()
    #return parser


if __name__ == "__main__":
    import sys
    import example_agent


    args = argument_proc(sys.argv[1:])
    #    agent = example_agent.ExampleAgent()
    import ddpg_agent
    agent = ddpg_agent.DDPGDriver()   
    C = Client(H=args.host, p=args.port, sid=args.id, e=args.episodes, t=args.track, s=args.stage, d=args.debug, agent=agent)
    C.setup_connection()
    C.run()
    C.shutdown()

