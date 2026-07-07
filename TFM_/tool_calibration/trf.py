from mecademicpy.robot import Robot

robot = Robot()
robot.Connect(address="192.168.0.100")

robot.ActivateRobot()
robot.Home()
robot.WaitHomed()

robot.SetTrf(-0.053, -0.457, 108.751, 0, 0, 0)
robot.WaitIdle()

pose = robot.GetPose()
print("Pose actual del flange:", pose)

robot.Disconnect()

