import numpy as np

class StanleyController:
    def __init__(self, k=1.0, wheel_base=2.5, max_steer=np.radians(30)):
        """
        Stanley lateral controller for path tracking.

        Parameters:
        - k: control gain (tuning parameter)
        - wheel_base: distance between front and rear axles (meters)
        - max_steer: maximum steering angle (radians)
        """
        self.k = k
        self.wheel_base = wheel_base
        self.max_steer = max_steer

        self.previous_steer = 0.0
        self.heading_gain = 0.5
        self.max_steer_change = 0.05

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def run(self, x, y, yaw, v, path_x, path_y, path_yaw, current_waypoint_idx):
        """
        Returns
        -------
        normalized steering command (-1 to 1)
        nearest waypoint index
        """

        front_x = x + self.wheel_base * 0.5 * np.cos(yaw)
        front_y = y + self.wheel_base * 0.5 * np.sin(yaw)

        dx = path_x - front_x
        dy = path_y - front_y
        dists = np.hypot(dx, dy)

        start = max(0, current_waypoint_idx - 10)
        end = min(len(path_x), current_waypoint_idx + 150)

        if end <= start:
            nearest_idx = int(current_waypoint_idx)
        else:
            local = int(np.argmin(dists[start:end]))
            nearest_idx = start + local

        # Calculate a smoother path heading using several waypoints
        # on either side of the nearest waypoint.
        heading_span = 5

        # Look farther ahead at high speed so the controller begins turning earlier.
        heading_lookahead = int(np.clip(v * 0.35, 5, 30))

        heading_center_idx = min(len(path_x) - 1, nearest_idx + heading_lookahead)

        previous_idx = max(0, heading_center_idx - heading_span)

        next_idx = min(len(path_x) - 1, heading_center_idx + heading_span)

        path_dx = path_x[next_idx] - path_x[previous_idx]
        path_dy = path_y[next_idx] - path_y[previous_idx]

        corrected_path_yaw = np.arctan2(path_dy, path_dx)

        heading_error = self.normalize_angle(corrected_path_yaw - yaw)

        error_x = front_x - path_x[nearest_idx]
        error_y = front_y - path_y[nearest_idx]

        cross_track_error = (-np.sin(corrected_path_yaw) * error_x + np.cos(corrected_path_yaw) * error_y)

        if abs(v) < 0.1:
            v = 0.1

        cte_term = np.arctan2(-self.k * cross_track_error, v)

        speed_scale = np.clip(40.0 / max(v, 1.0), 0.45, 1.0)

        raw_steer = -(self.heading_gain * speed_scale * heading_error + 0.3 * cte_term)

        steer = np.clip(raw_steer, -self.max_steer, self.max_steer)

        steer /= self.max_steer

        steer_change = np.clip(steer - self.previous_steer, -self.max_steer_change, self.max_steer_change)

        steer = self.previous_steer + steer_change
        self.previous_steer = steer

        return steer, nearest_idx
