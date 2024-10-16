# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from gym import spaces
import numpy as np
import torch
import omni.usd
from pxr import UsdGeom

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.cartpole import CartpoleTask
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole
from omniisaacgymenvs.robots.articulations.views.aloha_view import AlohaView
from omniisaacgymenvs.robots.articulations.jetbot import Jetbot

from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from pxr import PhysicsSchemaTools, UsdUtils, PhysxSchema, UsdPhysics
from pxr import Usd
from omni.physx import get_physx_simulation_interface
import omni.usd
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.prims import RigidContactView
from pxr import UsdGeom
CONTACT_FLAG = True
if CONTACT_FLAG:
    # from omni.isaac.core.utils.extensions import enable_extension
    # enable_extension("omni.isaac.range_sensor")
    from omni.isaac.sensor import _sensor
class CartpoleCameraTask(CartpoleTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        self._max_episode_length = 500

        self._num_observations = self.camera_width * self.camera_height * 3
        self._num_actions = 1
        self._jetbot_positions = torch.tensor([0.0, 0.0, 0.0])

        # use multi-dimensional observation for camera RGB
        self.observation_space = spaces.Box(
            np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * -np.Inf, 
            np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * np.Inf)

        RLTask.__init__(self, name, env)

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]

        self.camera_type = self._task_cfg["env"].get("cameraType", 'rgb')
        self.camera_width = self._task_cfg["env"]["cameraWidth"]
        self.camera_height = self._task_cfg["env"]["cameraHeight"]
        
        self.camera_channels = 3
        self._export_images = self._task_cfg["env"]["exportImages"]

    def cleanup(self) -> None:
        # initialize remaining buffers
        RLTask.cleanup(self)

        # override observation buffer for camera data
        self.obs_buf = torch.zeros(
            (self.num_envs, self.camera_width, self.camera_height, 3), device=self.device, dtype=torch.float)

    def add_camera(self) -> None:
        stage = get_current_stage()
        camera_path = f"/World/envs/env_0/Camera"
        camera_xform = stage.DefinePrim(f'{camera_path}_Xform', 'Xform')
        # set up transforms for parent and camera prims
        position = (-4.2, 0.0, 3.0)
        rotation = (0, -6.1155, -180)
        UsdGeom.Xformable(camera_xform).AddTranslateOp()
        UsdGeom.Xformable(camera_xform).AddRotateXYZOp()
        camera_xform.GetAttribute('xformOp:translate').Set(position)
        camera_xform.GetAttribute('xformOp:rotateXYZ').Set(rotation)
        camera = stage.DefinePrim(f'{camera_path}_Xform/Camera', 'Camera')
        UsdGeom.Xformable(camera).AddRotateXYZOp()
        camera.GetAttribute("xformOp:rotateXYZ").Set((90, 0, 90))
        # set camera properties
        camera.GetAttribute('focalLength').Set(24)
        camera.GetAttribute('focusDistance').Set(400)
        # hide other environments in the background
        camera.GetAttribute("clippingRange").Set((0.01, 20.0))

    def set_up_scene(self, scene) -> None:
        self.get_cartpole()
        self.add_camera()
        self.get_jetbot(scene)

        RLTask.set_up_scene(self, scene)

        # start replicator to capture image data
        self.rep.orchestrator._orchestrator._is_started = True

        # set up cameras
        self.render_products = []
        env_pos = self._env_pos.cpu()
        camera_paths = [f"/World/envs/env_{i}/Camera_Xform/Camera" for i in range(self._num_envs)]
        for i in range(self._num_envs):
            render_product = self.rep.create.render_product(camera_paths[i], resolution=(self.camera_width, self.camera_height))
            self.render_products.append(render_product)

        # initialize pytorch writer for vectorized collection
        self.pytorch_listener = self.PytorchListener()
        self.pytorch_writer = self.rep.WriterRegistry.get("PytorchWriter")
        self.pytorch_writer.initialize(listener=self.pytorch_listener, device="cuda")
        self.pytorch_writer.attach(self.render_products)

        self._cartpoles = ArticulationView(
            prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False
        )
        self._jetbots = AlohaView(prim_paths_expr="/World/envs/.*/aloha", name="jetbot_view")

        if CONTACT_FLAG:
            ## Training in static envs : use_flatcache = True (in task yaml file)
            ##_______________________________COLLISION
            
            self.my_stage = omni.usd.get_context().get_stage()
            self.my_prim = self.my_stage.GetPrimAtPath("/World/envs/env_0/aloha")
            
            from omni.isaac.sensor import ContactSensor
            self.sensor = ContactSensor(
                prim_path="/World/envs/env_0/aloha/base_link/Contact_Sensor",
                name="Contact_Sensor",
                frequency=60,
                translation=np.array([0, 0, 0]),
                min_threshold=0,
                max_threshold=10000000,
                radius=-1
            )
            contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(self.my_prim)
            contactReportAPI.CreateThresholdAttr().Set(20000)

        scene.add(self._cartpoles)
        scene.add(self._jetbots)
        return

    def get_observations(self) -> dict:
        dof_pos = self._cartpoles.get_joint_positions(clone=False)
        dof_vel = self._cartpoles.get_joint_velocities(clone=False)

        self.cart_pos = dof_pos[:, self._cart_dof_idx]
        self.cart_vel = dof_vel[:, self._cart_dof_idx]
        self.pole_pos = dof_pos[:, self._pole_dof_idx]
        self.pole_vel = dof_vel[:, self._pole_dof_idx]

        # retrieve RGB data from all render products
        images = self.pytorch_listener.get_rgb_data()
        if images is not None:
            if self._export_images:
                from torchvision.utils import save_image, make_grid
                img = images/255
                save_image(make_grid(img, nrow=2), '/home/kit/Music/cartpole_export.png')

            #self.obs_buf = torch.swapaxes(images, 1, 3).clone().float()/255.0
        else:
            print("Image tensor is NONE!")

        if CONTACT_FLAG:
            from pxr import PhysicsSchemaTools
            from omni.physx import get_physx_simulation_interface
            contact_headers, contact_data = get_physx_simulation_interface().get_contact_report()
            

            _contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
            contact_0 = _contact_sensor_interface.get_sensor_reading("/World/envs/env_0/aloha/base_link/Contact_Sensor", use_latest_data = True)
            print("0: ", contact_0.is_valid, contact_0.in_contact)
            contact_1 = _contact_sensor_interface.get_sensor_reading("/World/envs/env_1/aloha/base_link/Contact_Sensor", use_latest_data = True)
            print("1: ", contact_1.is_valid, contact_1.in_contact)
            contact_2 = _contact_sensor_interface.get_sensor_reading("/World/envs/env_2/aloha/base_link/Contact_Sensor", use_latest_data = True)
            print("2: ", contact_2.is_valid, contact_2.in_contact)
            contact_3 = _contact_sensor_interface.get_sensor_reading("/World/envs/env_3/aloha/base_link/Contact_Sensor", use_latest_data = True)
            print("3: ", contact_3.is_valid, contact_3.in_contact)
            
            value = self.sensor.get_current_frame()
            print("sens data: ", value)
            
            for contact_header in contact_headers:
                num_contact_data = contact_header.num_contact_data
                contact_data_offset = contact_header.contact_data_offset
                print("num", num_contact_data, contact_data_offset)
                # print("Got contact header type: " + str(contact_header.type))
                print("Actor0: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0)))
                print("Actor1: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1)))

        return self.obs_buf
