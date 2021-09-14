"A modified version of cartpole env using pybullet with collisions with an option to hide velocities"

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import pybullet as p
import os

dirname = os.path.dirname(__file__)
bulletfiles = os.path.join(dirname, 'cartpolebullet/')


class CartPoleBulletPOScaledEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts downright, and the goal is to get it upright over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to a modified version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle Sine           -1              1
        2	Pole Angle Cosine         -1              1       
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        See __init__ function for different reward function options.

    Episode Termination
        Cart Position is more than 4 (though there are elastic collisions at the boundaries, so not possible)
        Episode length is greater than 75
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        ### HYPER PARAMETERS FOR ENV ###
        self.INIT_BOTT = True       
        self.ALLOW_BOTT = True or self.INIT_BOTT
        self.TRIG_OBS = True
        self.REW_TOP_TOL = False
        self.REW_TOP_CONS = True and not(self.REW_TOP_TOL) ## For reward, keep it at the top for self.CONS_ROUNDS
        self.MAKE_MARKOVIAN = True and self.REW_TOP_CONS
        self.CART_VELOCITY_HIDDEN = True ## Works only for TRIG_OBS, MAKE_MARKOVIAN, REW_TOP_CONS= True for now
        self.POLE_VELOCITY_HIDDEN = True ## Works only for TRIG_OBS, MAKE_MARKOVIAN, REW_TOP_CONS= True for now
        self.TOP_TOL = 15*math.pi/180
        self.CONS_ROUNDS = 15
        self.reward_scale = 30.
        self.satisfied = 0
        self.work_on_angles = True
        
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.velMag = 15.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        if self.ALLOW_BOTT:
            self.theta_threshold_radians = 2 * math.pi
        else:
            self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 4 #2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
          
            
        high = np.array([self.x_threshold * 2])
        if self.CART_VELOCITY_HIDDEN:
            high = np.append(high, self.x_threshold * 2)
        else:
            high = np.append(high, np.finfo(np.float32).max)            
        
        if self.TRIG_OBS:
            high = np.append(high, 1)
            high = np.append(high, 1)
        else:
            high = np.append(high, self.theta_threshold_radians * 2)
            
        if self.POLE_VELOCITY_HIDDEN:
            high = np.append(high, 1)
            high = np.append(high, 1)
        else:
            high = np.append(high, np.finfo(np.float32).max)
       
        if self.MAKE_MARKOVIAN:
            high = np.append(high, 5000)
              
            
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        print("Pybullet Connected")
        world = p.connect(p.DIRECT)

        self.seed()
        self.viewer = None
        self.state = None
        self.previous_state = None
        self.done = False
        self.steps_beyond_done = None    
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        force = self.force_mag if action==1 else -self.force_mag
        targetVelocity = self.velMag if action==0 else -self.velMag
#         if self.MAKE_MARKOVIAN:
#             if self.TRIG_OBS:
#                 if self.CART_VELOCITY_HIDDEN:
#                     if self.POLE_VELOCITY_HIDDEN:
#                         x, x_prev, sintheta, costheta, sintheta_prev,costheta_prev,sat_round = state
#                     else:
#                         x, x_prev, sintheta, costheta, theta_dot,sat_round = state
#                 else:
#                     if self.POLE_VELOCITY_HIDDEN:
#                         x, x_dot, sintheta, costheta, sintheta_prev,costheta_prev,sat_round = state
#                     else:
#                         x, x_dot, sintheta, costheta, theta_dot,sat_round = state
#             else:
#                 x, x_dot, theta, theta_dot,sat_round = state
# #                 costheta = math.cos(theta)
# #                 sintheta = math.sin(theta)
#         else:
#             if self.TRIG_OBS:
#                 x, x_dot, sintheta, costheta, theta_dot = state
#             else:
#                 x, x_dot, theta, theta_dot = state
# #                 costheta = math.cos(theta)
# #                 sintheta = math.sin(theta)
            

        # Simulation
        p.setJointMotorControlArray(self.cp, [0], p.VELOCITY_CONTROL, 
                          targetVelocities=[targetVelocity], forces=[5000]) #exerts force of 5000 to reach target velocity
            
        if len(p.getContactPoints(self.cp, self.lb)) > 0:
        #simulates bouncing off of left block (simulation should do this on it's own, 
                    # but doesn't for some reason, maybe the way I'm doing velocity control)
            p.setJointMotorControlArray(self.cp, [0], p.VELOCITY_CONTROL, targetVelocities=[15], 
                                      forces=[p.getJointState(self.cp,0)[1]* 5000])
        elif len(p.getContactPoints(self.cp, self.rb)) > 0:
        #simulates bouncing off of right block (simulation should do this on it's own, 
                    # but doesn't for some reason, maybe the way I'm doing velocity control) 
            p.setJointMotorControlArray(self.cp, [0], p.VELOCITY_CONTROL, targetVelocities=[-15], 
                                      forces=[p.getJointState(self.cp,0)[1] * -5000])
       
        p.stepSimulation()
        self.state = p.getJointState(self.cp, 1)[0:2] + p.getJointState(self.cp, 0)[0:2]
        theta, theta_dot, x, x_dot = self.state
        
        

        #### Calculate next theta,sintheta, and costheta
#         if self.TRIG_OBS:
        sintheta = math.sin(theta)
        costheta = math.cos(theta)
                              
                
        if self.TRIG_OBS:
            if self.CART_VELOCITY_HIDDEN and (not self.POLE_VELOCITY_HIDDEN):
                self.state = (x,self.previous_state[0],sintheta,costheta,theta_dot)
            elif (not self.CART_VELOCITY_HIDDEN) and (self.POLE_VELOCITY_HIDDEN):
                self.state = (x,xdot,sintheta,costheta,self.previous_state[2],self.previous_state[3])
            elif self.CART_VELOCITY_HIDDEN and (self.POLE_VELOCITY_HIDDEN):
                self.state = (x,self.previous_state[0],sintheta,costheta,self.previous_state[2],self.previous_state[3])                
            else:
                self.state = (x,x_dot,sintheta,costheta,theta_dot)
            self.previous_state = self.state

        else:
            self.state = (x,x_dot,theta,theta_dot)
            
        ### Termination Criteria
        if self.ALLOW_BOTT:
#             done = False
            done = x < -self.x_threshold \
                    or x > self.x_threshold
        else:
            done =  theta < -self.theta_threshold_radians \
                    or theta > self.theta_threshold_radians
        self.done = bool(done)
        self.rew = 0
        if self.REW_TOP_TOL:
            if abs(sintheta) < math.sin(self.TOP_TOL) and costheta > math.cos(self.TOP_TOL):
                self.rew = 1.0
        elif self.REW_TOP_CONS:
            if abs(sintheta) < math.sin(self.TOP_TOL) and costheta > math.cos(self.TOP_TOL):
                self.satisfied += 1
                if self.satisfied >= self.CONS_ROUNDS: 
                    self.rew = 1.0
            else:
                self.satisfied = 0
        else:
            self.rew = 1.0
        if self.MAKE_MARKOVIAN:
            self.state = self.state + (self.satisfied,)
        if not self.done:
            reward = self.rew
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = self.rew
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        
        return np.array(self.state), reward/self.reward_scale, self.done, {}

    def reset(self):      
        p.resetSimulation()      
        self.cp = p.loadURDF(bulletfiles+"/cartpole.urdf")#, globalScaling=1.5)
        self.lb = p.loadURDF(bulletfiles+"/left-block.urdf", globalScaling=1.5)
        self.rb = p.loadURDF(bulletfiles+"/right-block.urdf", globalScaling=1.5)
        p.setPhysicsEngineParameter(restitutionVelocityThreshold=0.0)

        self.timeStep = 0.02
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)
        if self.TRIG_OBS:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(5,))
            if self.INIT_BOTT:
                self.state[3] -= 1 ## Cosine(pi/2)
            
            
            if self.state[2] > 0 and self.state[3] < 0:
                th = math.pi - math.asin(self.state[2])
            elif self.state[2] < 0 and self.state[3] < 0:
                th = -math.pi - math.asin(self.state[2])
            else:
                th = math.asin(self.state[2])
            
            p.resetJointState(self.cp, 1, th, self.state[4]) ## Pole
            p.resetJointState(self.cp, 0, self.state[0], self.state[1]) ## Cart         
            
        else:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
            
            p.resetJointState(self.cp, 1, self.state[2], self.state[3]) ## Pole
            p.resetJointState(self.cp, 0, self.state[0], self.state[1]) ## Cart               
        
        self.steps_beyond_done = None
            
        if self.CART_VELOCITY_HIDDEN:
            self.state[1] = self.state[0] - self.state[1]*self.timeStep
        if self.POLE_VELOCITY_HIDDEN:
            thetadot = self.state[4]
            self.state[4] = self.state[2] - thetadot*self.timeStep
            self.state = np.append(self.state,self.state[3] - thetadot*self.timeStep)
         
        if self.MAKE_MARKOVIAN:
            self.state = np.append(self.state,0)
            
        self.previous_state = self.state.copy()
        self.done = False
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        if self.TRIG_OBS:
            if x[2] > 0 and x[3] < 0:
                th = math.pi - math.asin(x[2])
            elif x[2] < 0 and x[3] < 0:
                th = -math.pi - math.asin(x[2])
            else:
                th = math.asin(x[2])
            self.poletrans.set_rotation(-th)
        else:
            self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        p.disconnect()
        print("Pybullet Disconnected")
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    def parameters_str(self):
#         angle = str("%.2f" % round(180*self.TOP_TOL/math.pi,2))
        angle = str(round(180*self.TOP_TOL/math.pi))
        return ("CartPolePOScaled : " + f"{self.reward_scale} ,"*(self.reward_scale != 30) +  
                "Hide Xdot, "*self.CART_VELOCITY_HIDDEN + 
                "Hide thetadot, "*self.POLE_VELOCITY_HIDDEN + 
                "Single Top Reward, "*self.REW_TOP_TOL +
               f"Cons:{self.CONS_ROUNDS}"*(self.REW_TOP_CONS and self.CONS_ROUNDS != 15))