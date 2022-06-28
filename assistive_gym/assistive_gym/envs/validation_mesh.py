import os, time
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .validation import ValidationEnv
from .agents import furniture
from .agents.furniture import Furniture

class ValidationMeshEnv(ValidationEnv):
    def __init__(self, robot, human):
        # super(FeedingMeshEnv, self).__init__(robot=robot, human=human)
        super(ValidationEnv, self).__init__(robot=robot, human=human, task='bed_bathing',
                                            obs_robot_len=(14 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)),
                                            obs_human_len=(15 + len(human.controllable_joint_indices)))
        self.general_model = True
        # Parameters for personalized human participants
        self.gender = 'male'
        self.body_shape= self.np_random.uniform(-2, 5, (1, self.human.num_body_shape)) #%s_1.pkl' % self.gender
        self.human_height = 1.8

    def step(self, action):
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action)

        obs = self._get_obs()

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.right_end_effector))

        spoon_pos, spoon_orient = self.tool.get_base_pos_orient()

        done = self.iteration >= 200
        reward = 1
        #done = 0
        info = 0

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {
                'robot': info, 'human': info}

    def get_total_force(self):
        robot_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        spoon_force_on_human = np.sum(self.tool.get_contact_points(self.human)[-1])
        return robot_force_on_human, spoon_force_on_human

    def get_food_rewards(self):
        # Check all food particles to see if they have left the spoon or entered the person's mouth
        # Give the robot a reward or penalty depending on food particle status
        food_reward = 0
        food_hit_human_reward = 0
        food_mouth_velocities = []
        foods_to_remove = []
        foods_active_to_remove = []
        for f in self.foods:
            food_pos, food_orient = f.get_base_pos_orient()
            distance_to_mouth = np.linalg.norm(self.target_pos - food_pos)
            if distance_to_mouth < 0.03:
                # Food is close to the person's mouth. Delete particle and give robot a reward
                food_reward += 20
                self.task_success += 1
                food_velocity = np.linalg.norm(f.get_velocity(f.base))
                food_mouth_velocities.append(food_velocity)
                foods_to_remove.append(f)
                foods_active_to_remove.append(f)
                f.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                continue
            elif len(f.get_closest_points(self.tool, distance=0.1)[-1]) == 0:
                # Delete particle and give robot a penalty for spilling food
                food_reward -= 5
                foods_to_remove.append(f)
                continue
        for f in self.foods_active:
            if len(f.get_contact_points(self.human)[-1]) > 0:
                # Record that this food particle just hit the person, so that we can penalize the robot
                food_hit_human_reward -= 1
                foods_active_to_remove.append(f)
        self.foods = [f for f in self.foods if f not in foods_to_remove]
        self.foods_active = [f for f in self.foods_active if f not in foods_active_to_remove]
        return food_reward, food_mouth_velocities, food_hit_human_reward

    def _get_obs(self, agent=None):
        spoon_pos, spoon_orient = self.tool.get_base_pos_orient()
        spoon_pos_real, spoon_orient_real = self.robot.convert_to_realworld(spoon_pos, spoon_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2 * np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        head_pos_real, head_orient_real = self.robot.convert_to_realworld(head_pos, head_orient)
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)
        self.robot_force_on_human, self.spoon_force_on_human = self.get_total_force()
        self.total_force_on_human = self.robot_force_on_human + self.spoon_force_on_human
        robot_obs = np.concatenate(
            [spoon_pos_real, spoon_orient_real, spoon_pos_real - target_pos_real, robot_joint_angles, head_pos_real,
             head_orient_real, [self.spoon_force_on_human]]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            spoon_pos_human, spoon_orient_human = self.human.convert_to_realworld(spoon_pos, spoon_orient)
            head_pos_human, head_orient_human = self.human.convert_to_realworld(head_pos, head_orient)
            target_pos_human, _ = self.human.convert_to_realworld(self.target_pos)
            human_obs = np.concatenate(
                [spoon_pos_human, spoon_orient_human, spoon_pos_human - target_pos_human, human_joint_angles,
                 head_pos_human, head_orient_human, [self.robot_force_on_human, self.spoon_force_on_human]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def reset(self):
        super(ValidationEnv, self).reset()
        self.build_assistive_env(furniture_type=None)
        #if self.robot.wheelchair_mounted:
        #wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
      #  self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]),
        #                                [0, 0, -np.pi / 2.0])


        if self.general_model:
            # Randomize the human body shape
            gender = self.np_random.choice(['male', 'female'])
            # body_shape = self.np_random.randn(1, self.human.num_body_shape)
            body_shape = self.np_random.uniform(-2, 5, (1, self.human.num_body_shape))
            # human_height = self.np_random.uniform(1.59, 1.91) if gender == 'male' else self.np_random.uniform(1.47, 1.78)
            human_height = self.np_random.uniform(1.5, 1.9)
        else:
            gender = self.gender
            body_shape = self.body_shape_filename
            human_height = self.human_height

        self.robot.set_base_pos_orient(np.array(self.robot.toc_base_pos_offset[self.task]),
                                       [0, 0, -np.pi / 2.0])

        # Update robot and human motor gains
        self.robot.motor_gains = self.human.motor_gains = 0.025

        joint_angles = [(self.human.j_left_hip_x, -90), (self.human.j_right_hip_x, -90), (self.human.j_left_knee_x, 70),
                        (self.human.j_right_knee_x, 70), (self.human.j_left_shoulder_z, -45),
                        (self.human.j_right_shoulder_z, 45), (self.human.j_left_elbow_y, -90),
                        (self.human.j_right_shoulder_y, 70), (self.human.j_right_shoulder_x, -60),
                        (self.human.j_right_elbow_y, 0)]

        # stretch out human arm
       # joints_positions = [(self.human.j_right_elbow, 0), (self.human.j_left_elbow, -90),
        #                        (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80),
       #                         (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        #joints_positions += [(self.human.j_head_x, 0),  # self.np_random.uniform(-30, 30)),
        #                         (self.human.j_head_y, 0),  # self.np_random.uniform(-30, 30)),
        #                         (self.human.j_head_z, 0)]  # self.np_random.uniform(-30, 30))]
       # joints_positions += [(self.human.j_right_shoulder_y, -90)]

        #self.human.set_base_pos_orient([0, -1, 0.95], [0, 0, -np.pi/2])
       # self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)

        self.human.init(self.directory, self.id, self.np_random, gender=gender, height=human_height,
                        body_shape=body_shape, joint_angles=joint_angles, position=[0, -1.1, 1.01], orientation=[0, 0, np.pi/2])

        # Create a table
        #self.table = Furniture()
        #self.table.init('table', self.directory, self.id, self.np_random)

        self.generate_target()

        p.resetDebugVisualizerCamera(cameraDistance=2.10, cameraYaw=40, cameraPitch=-45,
                                     cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=True,
                      mesh_scale=[0.1] * 3)

        target_ee_pos = np.array([-0.15, -0.65, 1.15]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        self.init_robot_pose(target_ee_pos, target_ee_orient,
                             [(target_ee_pos, target_ee_orient), (self.target_pos, None)],
                             [(self.target_pos, target_ee_orient)], arm='right', tools=[self.tool],
                             collision_objects=[self.human])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task],
                                             set_instantly=True)

        # Place a bowl on a table
       # self.bowl = Furniture()
        #self.bowl.init('bowl', self.directory, self.id, self.np_random)

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, 0)

        # p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Drop food in the spoon
        for _ in range(25):
            p.stepSimulation(physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    def generate_target(self):
        # Set target on mouth
        self.mouth_pos = [0, -0.11, -0.03] if self.human.gender == 'male' else [0, -0.1, 0.03]
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1],
                                                         physicsClientId=self.id)
        # init position on the human limb (forearm)
        target_pos = [-0.38, -0.84, 1.10]
        self.target = self.create_sphere(radius=0.01, mass=0.0, pos=target_pos, collision=False, rgba=[0, 1, 0, 1])
        self.update_targets()

    def update_targets(self):
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1],
                                                         physicsClientId=self.id)
        # init position on the human limb (forearm)
        self.target_pos = [-0.38, -0.84, 1.10]#np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, -0.11, -0.03])
