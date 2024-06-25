from Driver.simulator import LDSSimulation, DrivingSimulation, MujocoSimulation, FetchSimulation
import numpy as np


class LDS(LDSSimulation):
    def __init__(self, total_time=25, recording_time=[0,25]):
        super(LDS, self).__init__(name='lds', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 5
        self.state_size = 0
        self.feed_size = self.ctrl_size*self.input_size + self.state_size
        self.ctrl_bounds = [(-0.1,0.1),(-0.2,0.2),(-0.1,0.1),(-0.3,0.3),(-0.2,0.2)]*self.ctrl_size
        self.state_bounds = []
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 10

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)

        # speed (lower is better)
        speed1 = 3*np.mean(np.abs(recording[:,1])) / 0.47217050946
        speed2 = 3*np.mean(np.abs(recording[:,3])) / 0.14165115284
        speed3 = 3*np.mean(np.abs(recording[:,5])) / 0.70825576991
        speed4 = 3*np.mean(np.abs(recording[:,7])) / 0.47217051
        speed5 = 3*np.mean(np.abs(recording[:,9])) / 0.23608525

        # distance to the desired position (lower is better)
        distance1 = 3*np.mean(np.abs(recording[:,0]-1)) / 5.30746676867
        distance2 = 3*np.mean(np.abs(recording[:,2]-1)) / 1.16187855573
        distance3 = 3*np.mean(np.abs(recording[:,4]-1)) / 5.0294086958
        distance4 = 3*np.mean(np.abs(recording[:,6]-1)) / 1.38084352
        distance5 = 3*np.mean(np.abs(recording[:,8]-1)) / 1.13227806

        return [speed1, distance1, speed2, distance2, speed3, distance3, speed4, distance4, speed5, distance5]

    @property
    def state(self):
        return [self._state[i] for i in range(6)]
    @state.setter
    def state(self, value):
        self.reset()
        self.initial_state = value.copy()

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        interval_count = len(value)
        interval_time = int(self.total_time / interval_count)
        arr = np.array(arr).astype(float)
        for i in range(interval_count):
            arr[i*interval_time:(i+1)*interval_time] = value[i]
        self.ctrl = list(arr)

    def feed(self, value):
        ctrl_value = value[:]
        self.set_ctrl(ctrl_value)

    def get_cost_given_input(self, input):
        """

        :param input:
        :param weight:
        :return:
        """
        self.feed(list(input))
        features = np.array(self.get_features())
        return -np.dot(self.weights, features)  # minus as we want to maximize

    def find_optimal_path(self, weights):
        """
        New function to numerically find an optimal trajectory given weights
        Note: Using a generic numerical solver can lead to suboptimal solutions.
        :param weights:
        :param lb_input:
        :param ub_input:
        :return: optimal_controls, path_features, path_cost
        """
        from scipy.optimize import minimize
        self.weights = weights[0:self.num_of_features]
        lb_input = [x[0] for x in self.feed_bounds]
        ub_input = [x[1] for x in self.feed_bounds]
        random_start = [0] * self.feed_size
        bounds = np.transpose([lb_input, ub_input])
        res = minimize(self.get_cost_given_input, x0=random_start, bounds=bounds, method='L-BFGS-B')
        self.feed(list(res.x))
        features = np.array(self.get_features())
        controls = res.x
        return controls, features, -res.fun


class Driver(DrivingSimulation):
    """
    Original Driver model from 'Asking easy questions: A user-friendly approach to active reward learning'
    Bıyık, E., Palan, M., Landolfi, N. C., Losey, D. P., & Sadigh, D. (2019).arXiv preprint arXiv:1910.04365.
    """
    def __init__(self, total_time=50, recording_time=[0,50]):
        super(Driver ,self).__init__(name='driver', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 10
        self.state_size = 0
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = []
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 4

    def get_states(self):
        recording = self.get_recording(all_info=False)
        recording = list(recording)
        return recording

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)
        lb = [0.008, 0.031, 0, 0.622]
        lb = [0.0, 0.0, 0, 0.0]
        ub = [3, 1, 12, 0.956]

        # staying in lane (higher is better)
        not_in_lane = np.mean(np.exp(-30 * np.min(
            [np.square(recording[:, 0, 0] - 0.17), np.square(recording[:, 0, 0]), np.square(recording[:, 0, 0] + 0.17)],
            axis=0))) #/ 0.15343634
        # keeping speed (lower is better)
        speed_deviation = (np.mean(np.square(recording[:, 0, 3] - 1)) )#/ 0.42202643)
        # heading (higher is better)
        heading_error = np.mean(np.sin(recording[:, 0, 2])) #/ 0.06112367+4
        # collision avoidance (lower is better)
        closeness_to_other = (np.max(np.exp(-(7 * np.square(recording[:, 0, 0] - recording[:, 1, 0]) + 3 * np.square(
            recording[:, 0, 1] - recording[:, 1, 1])))) )#/ 0.15258019)
        # off_road = 10000 if np.max(recording[:, 0, 0]) > .26 or np.min(recording[:, 0, 0]) < -.26 else 0
        a = np.array(recording[:, 0, 0])
        # print(np.where(np.abs(a)>.26,a,len(a)*a))
        # print('WHERE',np.where(np.abs(a)>.26)[0])
        # print(len(np.where(np.abs(a)>.26)[0]))
        off_road = len(np.where(np.abs(a)>.25)[0]) / len(a)
        M = 1
        f = [1.-not_in_lane, speed_deviation, (1-heading_error)*10, closeness_to_other]
        f = np.subtract(f, lb)
        f = list(np.multiply(f, np.subtract(ub, lb)))# + [off_road]
        return f

    @property
    def state(self):
        return [self.robot.x, self.human.x]
    @state.setter
    def state(self, value):
        self.reset()
        self.initial_state = value.copy()

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        interval_count = len(value)//self.input_size
        interval_time = int(self.total_time / interval_count)
        arr = np.array(arr).astype(float)
        j = 0
        for i in range(interval_count):
            arr[i*interval_time:(i+1)*interval_time] = [value[j], value[j+1]]
            j += 2
        self.ctrl = list(arr)

    def feed(self, value):
        ctrl_value = value[:]
        self.set_ctrl(ctrl_value)

    def get_cost_given_input(self, input):
        """

        :param input:
        :param weight:
        :return:
        """
        self.feed(list(input))
        features = np.array(self.get_features())
        return -np.dot(self.weights, features)  #  no more minus - invert features instead!

    # def find_optimal_path(self, weights):
    #     """
    #     New function to numerically find an optimal trajectory given weights
    #     Note: Using a generic numerical solver can lead to suboptimal solutions.
    #     :param weights:
    #     :param lb_input:
    #     :param ub_input:
    #     :return: optimal_controls, path_features, path_cost
    #     """
    #     from scipy.optimize import minimize
    #     print("meee",self.num_of_features, weights, weights)
    #     self.weights = weights[0:self.num_of_features]
    #     lb_input = [x[0] for x in self.feed_bounds]
    #     ub_input = [x[1] for x in self.feed_bounds]
    #     random_start = [0] * self.feed_size
    #     # random_start = np.random.rand(self.feed_size)
    #     bounds = np.transpose([lb_input, ub_input])
    #     res = minimize(self.get_cost_given_input, x0=random_start, bounds=bounds, method='L-BFGS-B')
    #     self.feed(list(res.x))
    #     features = np.array(self.get_features())
    #     controls = res.x
    #     return controls, features, -res.fun

class DriverExtended(Driver):
    """
    Extended 10 dimensional driver model
    """
    def __init__(self, total_time=50, recording_time=[0,50]):
        super(Driver ,self).__init__(name='driverextended', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 10
        self.state_size = 0
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = []
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 10

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)
        # staying in lane (higher is better)
        staying_in_lane = np.mean(np.exp(-30*np.min([np.square(recording[:,0,0]-0.17), np.square(recording[:,0,0]), np.square(recording[:,0,0]+0.17)], axis=0))) / 0.15343634

        # keeping speed (lower is better)
        keeping_speed = -np.mean(np.square(recording[:,0,3]-1)) / 0.42202643

        # heading (higher is better)
        heading = np.mean(np.sin(recording[:,0,2])) / 0.06112367

        # collision avoidance (lower is better)
        collision_avoidance = -np.mean(np.exp(-(7*np.square(recording[:,0,0]-recording[:,1,0])+3*np.square(recording[:,0,1]-recording[:,1,1])))) / 0.15258019

        # min collision avoidance over time (lower is better)
        min_collision_avoidance = -np.max(np.exp(-(7*np.square(recording[:,0,0]-recording[:,1,0])+3*np.square(recording[:,0,1]-recording[:,1,1])))) / 0.10977646

        # average jerk (lower is better)
        acceleration = recording[1:,0,3] - recording[:-1,0,3]
        average_jerk = -np.mean(np.abs(acceleration[1:] - acceleration[:-1])) / 0.00317041

        # vertical displacement (higher is better)
        vertical_displacement = (recording[-1,0,1] - recording[0,0,1]) / 1.01818467


        final_left_lane = (recording[-1, 0, 0] > -.25) and (recording[-1, 0, 0] < -.09)
        final_right_lane = (recording[-1, 0, 0] > .09) and (recording[-1, 0, 0] < .25)
        final_center_lane = (recording[-1, 0, 0] > -.09) and (recording[-1, 0, 0] < .09)
        M=5
        return [staying_in_lane**2 + M,
                keeping_speed**2 + M,
                heading**2 + M,
                collision_avoidance**2 + M,
                min_collision_avoidance**2 + M,
                average_jerk**2 + M,
                vertical_displacement**2 + M,
                final_left_lane**2,
                final_right_lane**2,
                final_center_lane**2
                ]


class Tosser(MujocoSimulation):
    def __init__(self, total_time=1000, recording_time=[200,1000]):
        super(Tosser ,self).__init__(name='tosser', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 4
        self.state_size = 5
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = [(-0.2,0.2),(-0.785,0.785),(-0.1,0.1),(-0.1,-0.07),(-1.5,1.5)]
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 4

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)

        # horizontal range
        horizontal_range = -np.min([x[3] for x in recording]) / 0.25019166

        # maximum altitude
        maximum_altitude = np.max([x[2] for x in recording]) / 0.18554402

        # number of flips
        num_of_flips = np.sum(np.abs([recording[i][4] - recording[i-1][4] for i in range(1,len(recording))]))/(np.pi*2) / 0.33866545
        
        # distance to closest basket (gaussian fit)
        dist_to_basket = np.exp(-3*np.linalg.norm([np.minimum(np.abs(recording[len(recording)-1][3] + 0.9), np.abs(recording[len(recording)-1][3] + 1.4)), recording[len(recording)-1][2]+0.85])) / 0.17801466

        return [horizontal_range, maximum_altitude, num_of_flips, dist_to_basket]

    @property
    def state(self):
        return self.sim.get_state()
    @state.setter
    def state(self, value):
        self.reset()
        temp_state = self.initial_state
        temp_state.qpos[:] = value[:]
        self.initial_state = temp_state

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        arr[150:175] = [value[:self.input_size]]*25
        arr[175:200] = [value[self.input_size:2*self.input_size]]*25
        self.ctrl = arr

    def feed(self, value):
        initial_state = value[:self.state_size]
        ctrl_value = value[self.state_size:self.feed_size]
        self.initial_state.qpos[:] = initial_state
        self.set_ctrl(ctrl_value)



class Fetch(FetchSimulation):
    def __init__(self, total_time=1, recording_time=[0,1]):
        super(Fetch ,self).__init__(total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 1
        self.state_size = 0
        self.feed_size = self.ctrl_size + self.state_size
        self.num_of_features = 8

    def get_features(self):
        A = np.load('ctrl_samples/fetch.npz')
        return list(A['feature_set'][self.ctrl,:])

    @property
    def state(self):
        return 0
    @state.setter
    def state(self, value):
        pass

    def set_ctrl(self, value):
        self.ctrl = value

    def feed(self, value):
        self.set_ctrl(value)
