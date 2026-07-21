import matplotlib.pyplot as plt
import numpy as np
import roar_py_interface
from typing import List
from matplotlib.backend_bases import MouseButton

baseSection = []
replacementSection = []
findTarget = False
baseWaypoints = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load("competition_code\\waypoints\\waypointsPrimary.npz")
)

replacementWaypoints = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load("competition_code\\waypoints\\waypoints.npz")
)

newWaypoints = []

plt.figure(figsize=(12, 12))
plt.axis((-1100, 1100, -1100, 1100))
plt.tight_layout()
plt.ion()
plt.show()


def drawWaypoints(
    waypointsBase, waypointsNew=None, additionalWaypoints=baseWaypoints[:1]
):
    for i in waypointsBase:
        plt.plot(i.location[0], i.location[1], "ro")

    if waypointsNew != None:
        for i in waypointsNew:
            plt.plot(i.location[0], i.location[1], "co")

    for i in additionalWaypoints:
        plt.plot(i.location[0], i.location[1], "g^")

    plt.connect("button_press_event", on_click)


def distanceToWaypoint(currentLoc, waypoint: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(currentLoc[:2] - waypoint.location[:2])


def findClosestIndex(currentLoc, waypoints: List[roar_py_interface.RoarPyWaypoint]):
    lowestDist = 10
    closestInd = 0
    for i in range(0, len(waypoints)):
        dist = distanceToWaypoint(currentLoc, waypoints[i % len(waypoints)])
        if dist < lowestDist:
            lowestDist = dist
            closestInd = i
    return closestInd % len(waypoints)
    # return 0


def on_click(event):
    if event.button is MouseButton.LEFT:
        if len(baseSection) < 2:
            loc = [event.xdata, event.ydata]
            baseSection.append(findClosestIndex(loc, baseWaypoints))
            replacementSection.append(findClosestIndex(loc, replacementWaypoints))
            print(
                f"Plotting section waypoint {len(baseSection)} at ({loc[0]:.2f}, {loc[1]:.2f})"
            )
            plt.plot(event.xdata, event.ydata, "bs")
            plt.draw()
        else:
            plt.close()


print("\n--- Generating Waypoint Map ---\n")

drawWaypoints(baseWaypoints, replacementWaypoints)

while len(replacementSection) < 2:
    plt.pause(0.01)

print(
    f"Base waypoint indexes: {baseSection}\nLocations: {baseWaypoints[baseSection[0]].location}, {baseWaypoints[baseSection[1]].location}"
)
print(
    f"\nNew waypoint indexes: {replacementSection}\nLocations: {replacementWaypoints[replacementSection[0]].location}, {replacementWaypoints[replacementSection[1]].location}"
)

for i in range(baseSection[0]):
    newWaypoints.append(baseWaypoints[i])

if replacementSection[0] == replacementSection[1]:
    for i in range(replacementSection[0], len(replacementWaypoints)):
        newWaypoints.append(replacementWaypoints[i])
else:
    for i in range(replacementSection[0], replacementSection[1]):
        newWaypoints.append(replacementWaypoints[i])
    for i in range(baseSection[1], len(baseWaypoints)):
        newWaypoints.append(baseWaypoints[i])

print(f"\nWaypoints successfully spliced, saving as modifiedWaypoints.npz")
np.savez_compressed(
    "modifiedWaypoints.npz",
    **roar_py_interface.RoarPyWaypoint.save_waypoint_list(newWaypoints),
)
