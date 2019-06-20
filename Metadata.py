

class Metadata(object):
    def __init__(self, temp, boltzmann, viscosity, refractive_index, theta, lambda_0):
        self.temp = temp
        self.boltzmann_const = boltzmann
        self.viscosity = viscosity
        self.refractive_index = refractive_index
        self.theta = theta
        self.lambda_0 = lambda_0
