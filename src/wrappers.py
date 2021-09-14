# #         CARTPOLE
# #         x, Mean:0.45, std: 1.62
# #         xdot, Mean:0.23, std: 5.67
# #         sintheta, Mean:0.06, std: 0.56
# #         costheta, Mean:0.12, std: 0.82
# #         thetadot, Mean:-0.63, std: 4.39
# #         roundtop, Mean:5.09, std: 10.51
# #         action, Mean:0.51, std: 0.50

#     ACROBOT
#     cos(theta1), Mean:0.82, std: 0.32
#     sin(theta1), Mean:0.00, std: 0.48
#     cos(theta2), Mean:0.51, std: 0.59
#     sin(theta2), Mean:0.00, std: 0.63
#     thetaDot1, Mean:-0.01, std: 1.47
#     thetaDot2, Mean:-0.01, std: 2.69
#     Action, Mean:1.00, std: 1.74

#     LUNAR LANDER
#     x, Mean:-0.01, std: 0.32
#     y, Mean:0.9, std: 0.5
#     xdot, Mean:0.0, std: 0.62
#     ydot, Mean:-0.65, std: 0.5
#     theta, Mean:0.02, std: 0.5
#     thetaDot, Mean:0.01, std: 0.4
#     leg1, Mean:0.04, std: 0.2
#     leg2, Mean:0.04, std: 0.2
#     Action, Mean:1.50, std: 1.12


import numpy as np
class State_Normalizer(object):
    def __init__(self,env):
        try:                
            if "CartPoleBullet" in env.env.spec.id:
                print("Normalizing for CartPole")            
                if env.REW_TOP_TOL:
                    if env.CART_VELOCITY_HIDDEN:
                        if env.POLE_VELOCITY_HIDDEN:
                            self.normalize_state_inputs = self.normalize_6dim_xdot_thetadot
                        else:
                            self.normalize_state_inputs = self.normalize_5dim_xdot
                    else:
                        self.normalize_state_inputs = self.normalize_5dim
                else:
                    if env.CART_VELOCITY_HIDDEN:
                        if env.POLE_VELOCITY_HIDDEN:
                            self.normalize_state_inputs = self.normalize_7dim_xdot_thetadot
                        else:
                            self.normalize_state_inputs = self.normalize_6dim_xdot
                    else:
                        self.normalize_state_inputs = self.normalize_6dim  
            elif "Acrobot" in env.env.spec.id:
                print("Normalizing for Acrobot")
                self.normalize_state_inputs = self.normalize_acrobot           
            elif "Lunar" in env.env.spec.id:
                print("Normalizing for Lunar Lander") 
                if "Dummy" in env.env.spec.id:
                    self.normalize_state_inputs = self.lunar_lander_dummy_normalizer(env.env.n_dummy_observations,env.env.dummy_range)         
                else:    
                    self.normalize_state_inputs = self.normalize_lunar_lander
            else:
                self.normalize_state_inputs = self.identity
        except Exception as e:
            print(e)
            self.normalize_state_inputs = self.identity
            

            
    # This functions return appropriate normalizer for given number of dummy dimensions and ranges in Lunar Lander
    def lunar_lander_dummy_normalizer(self, dummy_dimensions, dummy_ranges):
        std = (2*dummy_ranges)/np.sqrt(12) 
        def normalize_lunar_lander(state):
            x, y, xdot, ydot, theta, thetadot, leg1, leg2 = state[:-dummy_dimensions]
            norm_state = np.array(((x+0.01)/0.32,(y-0.9)/0.5,(xdot)/0.62,(ydot+0.65)/0.5,
                                   (theta-0.02)/0.5,(thetadot-0.01)/0.4, (leg1-0.04)/0.2,(leg2-0.04)/0.2))
            
            dummy_states = np.array(state[-dummy_dimensions:])/std
   
            return np.concatenate([norm_state, dummy_states])
        return normalize_lunar_lander
            
            
    # This is for fully observable Lunar Lander env
    def normalize_lunar_lander(self,state):
        x, y, xdot, ydot, theta, thetadot, leg1, leg2 = state
        norm_state = np.array(((x+0.01)/0.32,(y-0.9)/0.5,(xdot)/0.62,(ydot+0.65)/0.5,
                               (theta-0.02)/0.5,(thetadot-0.01)/0.4, (leg1-0.04)/0.2,(leg2-0.04)/0.2))
        return norm_state            

    # This is for fully observable Acrobot env
    def normalize_acrobot(self,state):
        cos1, sin1,cos2,sin2, thetadot1, thetadot2 = state
        norm_state = np.array(((cos1-0.82)/0.32,(sin1)/0.48,(cos2-0.51)/0.59,(sin2)/0.63, thetadot1/1.47, thetadot2/2.69))
        return norm_state
    
    # This is for FULLY OBSERVABLE - REW_TOP_TOL env
    def normalize_5dim(self,state):
        x,xdot,sintheta,costheta,thetadot = state
        norm_state = np.array(((x-0.45)/1.62,(xdot-0.23)/5.67,sintheta/0.56,costheta/0.82,(thetadot+0.63)/4.4))
        return norm_state
                              
    # This is for FULLY OBSERVABLE - REW_TOP_CONS env
    def normalize_6dim(self,state):
        x,xdot,sintheta,costheta,thetadot,roundtop = state
        norm_state = np.array(((x-0.45)/1.62,(xdot-0.23)/5.67,sintheta/0.56,costheta/0.82,(thetadot+0.63)/4.4,(roundtop-5)/10.5))
        return norm_state
 
    # This is for XDOT HIDDEN - REW_TOP_CONS env
    def normalize_6dim_xdot(self,state):
        x,x_prev,sintheta,costheta,thetadot,roundtop = state
        norm_state = np.array(((x-0.45)/1.62,(x_prev-0.45)/1.62,sintheta/0.56,costheta/0.82,(thetadot+0.63)/4.4,(roundtop-5)/10.5))
        return norm_state
                              
    # This is for XDOT HIDDEN - REW_TOP_TOL env
    def normalize_5dim_xdot(self,state):
        x,x_prev,sintheta,costheta,thetadot = state
        norm_state = np.array(((x-0.45)/1.62,(x_prev-0.45)/1.62,sintheta/0.56,costheta/0.82,(thetadot+0.63)/4.4))
        return norm_state

    # This is for XDOT HIDDEN - THETADOT HIDDEN - REW_TOP_CONS env
    def normalize_7dim_xdot_thetadot(self,state):
        x,x_prev,sintheta,costheta,sintheta_prev,costheta_prev,roundtop = state
        norm_state = np.array(((x-0.45)/1.62,(x_prev-0.45)/1.62,sintheta/0.56,costheta/0.82,sintheta_prev/0.56,costheta_prev/0.82,(roundtop-5)/10.5))
        return norm_state
                              
    # This is for XDOT HIDDEN - THETADOT HIDDEN - REW_TOP_TOL env
    def normalize_6dim_xdot_thetadot(self,state):
        x,x_prev,sintheta,costheta,sintheta_prev,costheta_prev = state
        norm_state = np.array(((x-0.45)/1.62,(x_prev-0.45)/1.62,sintheta/0.56,costheta/0.82,sintheta_prev/0.56,costheta_prev/0.82))
        return norm_state                             
                  
    def identity(self,state):
        return state
                              
        
        
        
        
                              
# def normalize_state_inputs(state):
# #     if self.normalize_state_inputs_offline_metrics:
# #         x, Mean:0.45, std: 1.62
# #         xdot, Mean:0.23, std: 5.67
# #         sintheta, Mean:0.06, std: 0.56
# #         costheta, Mean:0.12, std: 0.82
# #         thetadot, Mean:-0.63, std: 4.39
# #         roundtop, Mean:5.09, std: 10.51
# #         action, Mean:0.51, std: 0.50
#       if (not self.CART_VELOCITY_HIDDEN) and (not self.POLE_VELOCITY_HIDDEN) 
# #             x,xdot,sintheta,costheta,thetadot,roundtop = state
#     x,x_prev,sintheta,costheta,sintheta_prev,costheta_prev,roundtop = state
# #             norm_state = np.array(((x-0.45)/1.62,(xdot-0.23)/5.67,sintheta/0.56,costheta/0.82,(thetadot+0.63)/4.4,(roundtop-5)/10.5))
# #             norm_state = np.array(((x-0.45)/1.62,(xdot-0.45)/1.62,sintheta/0.56,costheta/0.82,(thetadot+0.63)/4.4,(roundtop-5)/10.5))
#     norm_state = np.array(((x-0.45)/1.62,(x_prev-0.45)/1.62,sintheta/0.56,costheta/0.82,sintheta_prev/0.56,costheta_prev/0.82,(roundtop-5)/10.5))
#     return norm_state
# #     else:
# #         return state