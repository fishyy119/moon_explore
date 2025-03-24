# 纯纯是为了在没有ROS环境的系统欺骗Python，进而正常测试其他部分


class FakeQuaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class FakePoint:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class Pose:
    print("正在使用FakePose")

    def __init__(self):
        self.position = FakePoint()
        self.orientation = FakeQuaternion()
