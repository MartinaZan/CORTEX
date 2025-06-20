import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PhaseTrajectoryAnalyzer:
    def __init__(self, time, series, L=10, percentile=0.25):
        self.time = time
        self.series = series
        self.L = L
        self.percentile = percentile

        self.x, self.v, self.colors = self._compute_velocity_and_colors()
        self.idx_candidates, self.times_candidates = self._compute_low_distance_percentile()
        self.permanence = self._analyze_permanence()

    def get_idx_candidates_and_permanence(self):
        return self.idx_candidates, self.permanence

    def _compute_velocity_and_colors(self):
        dt_vec = np.diff(self.time)
        time_scaling = np.sum(dt_vec[:self.L])

        v = []
        colors = []

        for i in range(len(self.series) - self.L):
            segment = self.series[i:i+self.L+1]
            delta_x_sum = np.sum(np.abs(np.diff(segment)))
            dt_sum = np.sum(dt_vec[i:i+self.L]) / time_scaling * self.L
            v.append(delta_x_sum / dt_sum)

            if np.all(segment <= 0.5):
                colors.append('green')
            elif np.all(segment > 0.5):
                colors.append('red')
            else:
                colors.append('orange')

        return self.series[:-self.L], np.array(v), np.array(colors)

    def _compute_low_distance_percentile(self):
        distances = np.sqrt(self.x**2 + self.v**2)
        threshold = np.quantile(distances, self.percentile)
        idx_candidates = np.where(distances <= threshold)[0]
        times_candidates = self.time[idx_candidates]

        return idx_candidates, times_candidates
    
    def _analyze_permanence(self):
        idx_candidates = self.idx_candidates
        idx_candidates = np.sort(idx_candidates)
        set_candidates = set(idx_candidates)
        permanence = []
        for t in idx_candidates:
            tau = 0
            while (t + 1 + tau) in set_candidates:
                tau += 1
            permanence.append(tau)
        
        return np.array(permanence)

    def plot_phase_trajectory_final(self):
        fig = plt.figure(figsize=(4, 4))
        ax_phase = fig.add_subplot(1, 1, 1)
        ax_phase.set_xlabel('x(t)')
        ax_phase.set_ylabel(f'v(t) with lag {self.L}')
        ax_phase.set_title('Phase trajectory')
        ax_phase.set_xlim(-0.05, 0.55)
        ax_phase.set_ylim(-0.05, 0.55)
        ax_phase.scatter(self.x[self.colors == 'green'], self.v[self.colors == 'green'],
                         color='green', s=10, alpha=0.6, zorder=5)
        ax_phase.scatter(self.x[self.colors == 'red'], self.v[self.colors == 'red'],
                         color='red', s=10, alpha=0.6, zorder=5)
        ax_phase.scatter(self.x[self.colors == 'orange'], self.v[self.colors == 'orange'],
                         color='orange', s=10, alpha=0.6, zorder=5)
        ax_phase.plot(self.x, self.v, color='black', alpha=0.4,
                      linestyle='dashed', linewidth=1, zorder=1)
        plt.tight_layout()
        plt.show()

    # def plot_low_distance_percentile(self):
    #     print(f"Number of points below the {self.percentile*100}% percentile: {len(self.times_candidates)}")
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(self.time, self.series, alpha=0.7, label='x(t)')
    #     plt.plot(self.times_candidates, self.series[self.idx_candidates], marker='.', color='green', linestyle='None', markersize=5, label=f'Points <= {self.percentile*100} percentile')
    #     plt.xlabel('time')
    #     plt.ylabel('x(t)')
    #     plt.legend()
    #     plt.show()

    # def plot_hist_permanence(self):
    #     permanence = self.permanence
    #     print(f"Average duration of consecutive permanence in C: {permanence.mean():.2f}")
    #     print(f"Min duration: {permanence.min()}")
    #     print(f"Max duration: {permanence.max()}")
    #     plt.figure(figsize=(6, 3))
    #     plt.hist(permanence, bins=permanence.max())
    #     plt.xlabel("Consecutive duration")
    #     plt.ylabel("Frequency")
    #     plt.title("Histogram of consecutive permanence")
    #     plt.show()

    def plot_time_series_with_permanence(self):
        idx_candidates = self.idx_candidates
        permanence = self.permanence

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()
        dt = np.median(np.diff(self.time[idx_candidates])) if len(idx_candidates) > 1 else 1
        ax2.bar(self.time[idx_candidates], permanence, width=dt, color='seagreen', alpha=0.2)
        ax2.set_ylabel(r"Permanence time $\tau$", color='seagreen')
        ax2.set_ylim(-1, permanence.max() * 1.1)
        ax2.tick_params(axis='y', labelcolor='seagreen')
        ax1.plot(self.time, self.series, alpha=0.7, label='x(t)')
        ax1.plot(self.time[idx_candidates], self.series[idx_candidates], '.', color='green', markersize=4)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("x(t)", color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        plt.show()
