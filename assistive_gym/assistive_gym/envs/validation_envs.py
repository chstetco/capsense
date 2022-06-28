from .validation import ValidationEnv
from .validation_mesh import ValidationMeshEnv
from .agents import pr2, baxter, sawyer, jaco, stretch, panda, human
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.panda import Panda
from .agents.human import Human
from .agents.human_mesh import HumanMesh
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm = 'left'
human_controllable_joint_indices = human.left_arm_joints
class ValidationPR2Env(ValidationEnv):
    def __init__(self):
        super(ValidationPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ValidationBaxterEnv(ValidationEnv):
    def __init__(self):
        super(ValidationBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ValidationSawyerEnv(ValidationEnv):
    def __init__(self):
        super(ValidationSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ValidationJacoEnv(ValidationEnv):
    def __init__(self):
        super(ValidationJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ValidationStretchEnv(ValidationEnv):
    def __init__(self):
        super(ValidationStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ValidationPandaEnv(ValidationEnv):
    def __init__(self):
        super(ValidationPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ValidationPR2HumanEnv(ValidationEnv, MultiAgentEnv):
    def __init__(self):
        super(ValidationPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ValidationPR2Human-v1', lambda config: ValidationPR2HumanEnv())

class ValidationBaxterHumanEnv(ValidationEnv, MultiAgentEnv):
    def __init__(self):
        super(ValidationBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ValidationBaxterHuman-v1', lambda config: ValidationBaxterHumanEnv())

class ValidationSawyerHumanEnv(ValidationEnv, MultiAgentEnv):
    def __init__(self):
        super(ValidationSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ValidationSawyerHuman-v1', lambda config: ValidationSawyerHumanEnv())

class ValidationJacoHumanEnv(ValidationEnv, MultiAgentEnv):
    def __init__(self):
        super(ValidationJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ValidationJacoHuman-v1', lambda config: ValidationJacoHumanEnv())

class ValidationStretchHumanEnv(ValidationEnv, MultiAgentEnv):
    def __init__(self):
        super(ValidationStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ValidationStretchHuman-v1', lambda config: ValidationStretchHumanEnv())

class ValidationPandaHumanEnv(ValidationEnv, MultiAgentEnv):
    def __init__(self):
        super(ValidationPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ValidationPandaHuman-v1', lambda config: ValidationPandaHumanEnv())

class ValidationPR2MeshEnv(ValidationMeshEnv):
    def __init__(self):
        super(ValidationR2MeshEnv, self).__init__(robot=PR2(robot_arm), human=HumanMesh())

class ValidationBaxterMeshEnv(ValidationMeshEnv):
    def __init__(self):
        super(ValidationBaxterMeshEnv, self).__init__(robot=Baxter(robot_arm), human=HumanMesh())

class ValidationSawyerMeshEnv(ValidationMeshEnv):
    def __init__(self):
        super(ValidationSawyerMeshEnv, self).__init__(robot=Sawyer(robot_arm), human=HumanMesh())

class ValidationJacoMeshEnv(ValidationMeshEnv):
    def __init__(self):
        super(ValidationJacoMeshEnv, self).__init__(robot=Jaco(robot_arm), human=HumanMesh())

class ValidationStretchMeshEnv(ValidationMeshEnv):
    def __init__(self):
        super(ValidationStretchMeshEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=HumanMesh())

class ValidationPandaMeshEnv(ValidationMeshEnv):
    def __init__(self):
        super(ValidationPandaMeshEnv, self).__init__(robot=Panda(robot_arm), human=HumanMesh())
