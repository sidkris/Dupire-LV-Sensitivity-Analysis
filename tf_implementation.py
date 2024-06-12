import tensorflow as tf
import numpy as np

class DupireLocalVolatilityModel:
    def __init__(self, initial_price, volatility_surface, time_grid, price_grid):
        self.initial_price = initial_price
        self.volatility_surface = volatility_surface
        self.time_grid = time_grid
        self.price_grid = price_grid

    def interpolate_volatility(self, price, time):
        # Bilinear interpolation for the local volatility surface
        return tf.image.resize(self.volatility_surface, [price, time], method='bilinear')

    def simulate_path(self, num_steps, dt, num_paths):
        paths = np.zeros((num_paths, num_steps))
        paths[:, 0] = self.initial_price
        for t in range(1, num_steps):
            vol = self.interpolate_volatility(paths[:, t-1], self.time_grid[t])
            drift = 0  # assuming drift is 0 for simplicity
            diffusion = vol * np.random.normal(size=num_paths) * np.sqrt(dt)
            paths[:, t] = paths[:, t-1] * (1 + drift * dt + diffusion)
        return paths

class EuropeanCallOption:
    def __init__(self, strike_price, maturity):
        self.strike_price = strike_price
        self.maturity = maturity

    def payoff(self, paths):
        return np.maximum(paths[:, -1] - self.strike_price, 0)

class MonteCarloSimulator:
    def __init__(self, model, option, num_paths, num_steps, dt):
        self.model = model
        self.option = option
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = dt

    def estimate_option_price(self):
        paths = self.model.simulate_path(self.num_steps, self.dt, self.num_paths)
        payoffs = self.option.payoff(paths)
        return np.mean(payoffs) * np.exp(-self.model.interest_rate * self.option.maturity)

    def estimate_sensitivities(self):
        with tf.GradientTape() as tape:
            paths = self.model.simulate_path(self.num_steps, self.dt, self.num_paths)
            payoffs = self.option.payoff(paths)
            option_price = np.mean(payoffs) * np.exp(-self.model.interest_rate * self.option.maturity)
        gradients = tape.gradient(option_price, self.model.volatility_surface)
        return gradients

class TPUSimulator:
    def __init__(self, model, option, num_paths, num_steps, dt):
        self.model = model
        self.option = option
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = dt

    def run(self):
        strategy = tf.distribute.experimental.TPUStrategy()
        with strategy.scope():
            simulator = MonteCarloSimulator(self.model, self.option, self.num_paths, self.num_steps, self.dt)
            option_price = simulator.estimate_option_price()
            sensitivities = simulator.estimate_sensitivities()
        return option_price, sensitivities

def main():
    # Define model parameters
    initial_price = 100
    strike_price = 105
    maturity = 1.0
    num_paths = 500000
    num_steps = 100
    dt = maturity / num_steps

    # Define volatility surface (dummy data for example purposes)
    time_grid = np.linspace(0, maturity, num_steps)
    price_grid = np.linspace(50, 150, num_steps)
    volatility_surface = np.random.uniform(low=0.1, high=0.3, size=(num_steps, num_steps))

    # Instantiate the model and option
    model = DupireLocalVolatilityModel(initial_price, volatility_surface, time_grid, price_grid)
    option = EuropeanCallOption(strike_price, maturity)

    # Run simulation on TPU
    tpu_simulator = TPUSimulator(model, option, num_paths, num_steps, dt)
    option_price, sensitivities = tpu_simulator.run()

    print("Estimated Option Price:", option_price)
    print("Sensitivities:", sensitivities)

if __name__ == "__main__":
    main()
