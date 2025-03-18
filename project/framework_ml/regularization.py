from framework_ml.puromachine import *

class Regularization():
    @staticmethod
    def add_l_regularization(regularization, regularization_term, s_L, s_weights):
        if regularization == 'l1':
            s_p=Some.function(s_weights, AbsoluteValueFunction)
            s_q=Some.function(s_p, SumFunction)
            s_r=Some.scalar_mul(regularization_term, s_q)
            s_L=Some.add(s_L, s_r)
        if regularization == 'l2':
            s_p=Some.function(s_weights, PowerFunction, b=2)
            s_q=Some.function(s_p, SumFunction)
            s_r=Some.scalar_mul(regularization_term / 2, s_q)
            s_L=Some.add(s_L, s_r)
        return s_L