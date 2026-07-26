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
        window = 150

        start = max(0, current_waypoint_idx - 10)
        end = min(len(path_x), current_waypoint_idx + 150)

        local = np.argmin(dists[start:end])
        nearest_idx = start + local

        heading_error = self.normalize_angle(
            path_yaw[nearest_idx] - yaw
        )
        # If the path tangent points backward, reverse it
        if abs(heading_error) > np.pi / 2:
            corrected_path_yaw = self.normalize_angle(
                path_yaw[nearest_idx] + np.pi
            )

            heading_error = self.normalize_angle(
                corrected_path_yaw - yaw
            )
        else:
            corrected_path_yaw = path_yaw[nearest_idx]

        error_x = front_x - path_x[nearest_idx]
        error_y = front_y - path_y[nearest_idx]

        cross_track_error = (
            -np.sin(corrected_path_yaw) * error_x
            + np.cos(corrected_path_yaw) * error_y
        )

        if abs(v) < 0.1:
            v = 0.1

        cte_term = np.arctan2(
            -self.k * cross_track_error,
            v,
        )

        steer = heading_error

        steer = np.clip(
            steer,
            -self.max_steer,
            self.max_steer,
        )

        steer /= self.max_steer

        return steer, nearest_idx