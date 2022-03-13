import numpy as np
import matplotlib.pyplot as plt


class Parabolic:
    def __init__(self, diffusivity, end_time, xlim, u0, border_cond):
        self.a = diffusivity
        self.dt, self.dx = 0.001, 0.001
        self.T = np.linspace(0, end_time, end_time * 100 + 1)
        a, b = xlim
        self.X = np.linspace(a, b, 101)
        self.left, self.right = border_cond
        self.c = 0.26
        self.A, self.A_inv = self.calc_A()
        self.u0 = u0
        self.solution = self.solve_eq()

    def calc_A(self):
        n = len(self.X)
        line = np.array([-self.c * self.dt * self.a / self.dx ** 2, 1 + 2 * self.c * self.a * self.dt / self.dx ** 2,
                         -self.c * self.dt * self.a / self.dx ** 2])
        A = [[1] + [0] * (n - 1)]
        for i in range(n - 2):
            A_line = np.zeros(n)
            A_line[i:i + 3] = line
            A.append(A_line)
        A.append([0] * (n - 1) + [1])
        print(np.vstack(A))
        return np.vstack(A), np.linalg.inv(A)

    def right_part(self, t_next, cur):
        cur_u_l = cur[:-2]
        cur_u_c = cur[1:-1]
        cur_u_r = cur[2:]
        B_middle = cur_u_c + (1 - self.c) * self.a * self.dt * (cur_u_r - 2 * cur_u_c + cur_u_l) / self.dx ** 2
        B = np.concatenate([[self.left(t_next)], B_middle, [self.right(t_next)]])
        return B

    @staticmethod
    def solve_sys(A_inv, B):
        return A_inv @ B

    def solve_eq(self):
        u0 = self.u0(self.X)
        solution = [u0]
        for t in self.T[2:]:
            B = self.right_part(t, solution[-1])
            _cur = self.solve_sys(self.A_inv, B)
            solution.append(_cur)
        return solution

    def plot_solution(self, t, ylim):
        idx = (np.abs(self.T - t)).argmin()
        calculated = self.solution[idx]

        font = {'weight': 'bold', 'size': 10}
        plt.rc('font', **font)
        plt.figure(figsize=(10, 5))
        plt.plot(self.X, calculated)
        plt.title(f'Solution {np.round(self.T[idx], 1)}')
        plt.ylim(ylim[0], ylim[1])
        return plt.show()
