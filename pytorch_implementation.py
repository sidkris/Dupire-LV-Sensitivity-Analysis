import torch
import numpy as np

class DupireLocalVolatilityModel:
    def __init__(self, initial_price, volatility_surface, time_grid, price_grid):
        self.initial_price = initial_price
        self.volatility_surface = torch.tensor(volatility_surface, dtype=torch.float32)
        self.time_grid = torch.tensor(time_grid, dtype=torch.float32)
        self.price_grid = torch.tensor(price_grid, dtype=torch.float32)

    def interpolate_volatility(self, price, time):
        price = torch.clamp(price, self.price_grid.min(), self.price_grid.max())
        time = torch.clamp(time, self.time_grid.min(), self.time_grid.max())
        price_idx = torch.searchsorted(self.price_grid, price) - 1
        time_idx = torch.searchsorted(self.time_grid, time) - 1
        price_idx = torch.clamp(price_idx, 0, self.volatility_surface.shape[0] - 2)
        time_idx = torch.clamp(time_idx, 0, self.volatility_surface.shape[1] - 2)
        
        # Interpolation
        vol_tl = self.volatility_surface[price_idx, time_idx]
        vol_tr = self.volatility_surface[price_idx, time_idx + 1]
        vol_bl = self.volatility_surface[price_idx + 1, time_idx]
        vol_br = self.volatility_surface[price_idx + 1, time_idx + 1]

        price_frac = (price - self.price_grid[price_idx]) / (self.price_grid[price_idx + 1] - self.price_grid[price_idx])
        time_frac = (time - self.time_grid[time_idx]) / (self.time_grid[time_idx + 1] - self.time_grid[time_idx])

        vol = vol_tl * (1 - price_frac) * (1 - time_frac) + vol_tr * time_frac * (1 - price_frac) + vol_bl * price_frac * (1 - time_frac) + vol_br * price_frac * time_frac

        return vol

    def simulate_path(self, num_steps, dt, num_paths):
        paths = torch.zeros((num_paths, num_steps), dtype=torch.float32)
        paths[:, 0] = self.initial_price
        for t in range(1, num_steps):
            vol = self.interpolate_volatility(paths[:, t-1], self.time_grid[t])
            drift = 0  # assuming drift is 0 for simplicity
            diffusion = vol * torch.randn(num_paths) * torch.sqrt(torch.tensor(dt, dtype=torch.float32))
            paths[:, t] = paths[:, t-1] * (1 + drift * dt + diffusion)
        return paths

class EuropeanCallOption:
    def __init__(self, strike_price, maturity):
        self.strike_price = strike_price
        self.maturity = maturity

    def payoff(self, paths):
        return torch.clamp(paths[:, -1] - self.strike_price, min=0)

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
        return torch.mean(payoffs) * torch.exp(-self.model.interest_rate * self.option.maturity)

    def estimate_sensitivities(self):
        paths = self.model.simulate_path(self.num_steps, self.dt, self.num_paths)
        payoffs = self.option.payoff(paths)
        option_price = torch.mean(payoffs) * torch.exp(-self.model.interest_rate * self.option.maturity)
        option_price.backward()
        return self.model.volatility_surface.grad

class GPUSimulator:
    def __init__(self, model, option, num_paths, num_steps, dt):
        self.model = model
        self.option = option
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = dt

    def run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.volatility_surface = self.model.volatility_surface.to(device).requires_grad_()
        simulator = MonteCarloSimulator(self.model, self.option, self.num_paths, self.num_steps, self.dt)
        option_price = simulator.estimate_option_price()
        sensitivities = simulator.estimate_sensitivities()
        return option_price.item(), sensitivities.cpu().detach().numpy()

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

    # Run simulation on GPU
    gpu_simulator = GPUSimulator(model, option, num_paths, num_steps, dt)
    option_price, sensitivities = gpu_simulator.run()

    print("Estimated Option Price:", option_price)
    print("Sensitivities:", sensitivities)

if __name__ == "__main__":
    main()
