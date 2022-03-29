import glob
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)
import gym
import gym.spaces
import gym.utils
import gym.utils.seeding
import numpy as np
import torch
import torch.nn.functional as F
from common.misc_utils import line_to_point_distance
from environments.mocap_renderer import extract_joints_xyz


FOOT2METER = 0.3048
METER2FOOT = 1 / 0.3048


class EnvBase(gym.Env):
    def __init__(
        self,
        num_parallel,
        device,
        pose_vae_path,
        rendered=False,
        use_params=False,
        camera_tracking=True,
        frame_skip=1,
    ):
        self.np_random = None
        self.seed()

        self.is_rendered = rendered
        self.num_parallel = num_parallel
        self.frame_skip = frame_skip
        self.device = device

        self.load_data(pose_vae_path)

        self.action_scale = 4.0
        self.data_fps = 30
        self.frame_dim = self.mocap_data.shape[1]
        self.num_condition_frames = self.pose_vae_model.num_condition_frames
        # action_dim is the latent dim
        self.action_dim = (
            self.pose_vae_model.latent_size
            if not hasattr(self.pose_vae_model, "quantizer")
            else self.pose_vae_model.quantizer.num_embeddings
        )

        self.max_timestep = int(1200 / self.frame_skip)

        # history size is used to calculate floating as well
        self.history_size = 5
        assert (
            self.history_size >= self.num_condition_frames
        ), "History size has to be greater than condition size."
        self.history = torch.zeros(
            (self.num_parallel, self.history_size, self.frame_dim)
        ).to(self.device)

        indices = self.np_random.randint(0, self.mocap_data.shape[0], self.num_parallel)
        indices = torch.from_numpy(indices).long()

        self.start_indices = indices
        self.root_facing = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.root_xz = torch.zeros((self.num_parallel, 2)).to(self.device)
        self.reward = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.potential = torch.zeros((self.num_parallel, 2)).to(self.device)
        self.done = torch.zeros((self.num_parallel, 1)).bool().to(self.device)

        # used for reward-based early termination
        self.parallel_ind_buf = (
            torch.arange(0, self.num_parallel).long().to(self.device)
        )

        # 4 and 7 are height for right and left toes respectively
        # y-axis in the data, but z-axis in the env
        self.foot_xy_ind = torch.LongTensor([[15, 17], [60, 62]])  ## left, right toe
        self.foot_z_ind = torch.LongTensor([16, 61])
        self.contact_threshold = 0.03 * METER2FOOT
        self.foot_pos_history = torch.zeros((self.num_parallel, 2, 6)).to(self.device)

        indices = torch.arange(0, 96).long().to(self.device)
        x_indices = indices[slice(3, 96, 3)]
        y_indices = indices[slice(4, 96, 3)]
        z_indices = indices[slice(5, 96, 3)]
        self.joint_indices = (x_indices, y_indices, z_indices)

        if self.is_rendered:
            from .custom_renderer import CustomMocapViewer

            self.viewer = CustomMocapViewer(
                self,
                num_characters=num_parallel,
                x_ind=x_indices,
                y_ind=y_indices,
                z_ind=z_indices,
                target_fps=self.data_fps,
                use_params=use_params,
                camera_tracking=camera_tracking,
            )

        high = np.inf * np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def load_data(self, pose_vae_path):
        # mocap_file = os.path.join(current_dir, "dance_pose0.npy")
        # mocap_file = os.path.join(current_dir, "pose_351.npy")
        print(current_dir)
        mocap_file = os.path.join(current_dir, "dance_test.npy")
        data = torch.from_numpy(np.load(mocap_file))
        self.mocap_data = data.float().to(self.device)

        if os.path.isdir(pose_vae_path):
            basepath = os.path.normpath(pose_vae_path)
            pose_vae_path = glob.glob(os.path.join(basepath, "posevae*.pt"))[0]
        else:
            basepath = os.path.dirname(pose_vae_path)

        self.pose_vae_model = torch.load(pose_vae_path, map_location=self.device)
        self.pose_vae_model.eval()

        assert (
            self.pose_vae_model.num_future_predictions >= self.frame_skip
        ), "VAE cannot skip this many frames"

        print("=========")
        print("Loaded: ", mocap_file)
        print("Loaded: ", pose_vae_path)
        print("=========")

    def integrate_root_translation(self, pose):
        mat = self.get_rotation_matrix(self.root_facing)
        displacement = (mat * pose[:, 0:2].unsqueeze(1)).sum(dim=2)
        self.root_facing.add_(pose[:, [2]]).remainder_(2 * np.pi)
        self.root_xz.add_(displacement)

        self.history = self.history.roll(1, dims=1)
        self.history[:, 0].copy_(pose)

        foot_z = pose[:, self.foot_z_ind].unsqueeze(-1)
        foot_xy = pose[:, self.foot_xy_ind]
        foot_pos = torch.cat((self.root_xz.unsqueeze(1) + foot_xy, foot_z), dim=-1)
        self.foot_pos_history = self.foot_pos_history.roll(1, dims=1)
        self.foot_pos_history[:, 0].copy_(foot_pos.flatten(1, 2))

    def get_rotation_matrix(self, yaw, dim=2):
        yaw = -yaw
        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)
        if dim == 3:
            col1 = torch.cat((yaw.cos(), yaw.sin(), zeros), dim=-1)
            col2 = torch.cat((-yaw.sin(), yaw.cos(), zeros), dim=-1)
            col3 = torch.cat((zeros, zeros, ones), dim=-1)
            matrix = torch.stack((col1, col2, col3), dim=-1)
        else:
            col1 = torch.cat((yaw.cos(), yaw.sin()), dim=-1)
            col2 = torch.cat((-yaw.sin(), yaw.cos()), dim=-1)
            matrix = torch.stack((col1, col2), dim=-1)
        return matrix

    def get_vae_condition(self, normalize=False, flatten=True):
        condition = self.history[:, : self.num_condition_frames]
        if normalize:
            condition = self.pose_vae_model.normalize(condition)
        if flatten:
            condition = condition.flatten(start_dim=1, end_dim=2)
        return condition

    def get_vae_next_frame(self, action):
        self.action = action
        condition = self.get_vae_condition(normalize=True, flatten=False)

        with torch.no_grad():
            condition = condition.flatten(start_dim=1, end_dim=2)
            vae_output = self.pose_vae_model.sample(
                action, condition, deterministic=True
            )
            vae_output = vae_output.view(
                -1,
                self.pose_vae_model.num_future_predictions,
                self.pose_vae_model.frame_size,
            )
        next_frame = self.pose_vae_model.denormalize(vae_output)
        return next_frame

    def reset_initial_frames(self, frame_index=None):
        # Make sure condition_range doesn't blow up
        self.start_indices.random_(
            0, self.mocap_data.shape[0] - self.num_condition_frames + 1
        )

        if self.is_rendered:
            # controlled from GUI
            param_name = "debug_frame_index"
            if hasattr(self, param_name) and getattr(self, param_name) != -1:
                self.start_indices.fill_(getattr(self, param_name))

        # controlled from CLI
        if frame_index is not None:
            if self.is_rendered:
                setattr(self, param_name, frame_index)
                self.start_indices.fill_(getattr(self, param_name))
            else:
                self.start_indices.fill_(frame_index)

        # Newer has smaller index (ex. index 0 is newer than 1)
        condition_range = (
            self.start_indices.repeat((self.num_condition_frames, 1)).t()
            + torch.arange(self.num_condition_frames - 1, -1, -1).long()
        )

        self.history[:, : self.num_condition_frames].copy_(
            self.mocap_data[condition_range]
        )

    def calc_foot_slide(self):
        foot_z = self.foot_pos_history[:, :, [2, 5]]
        # in_contact = foot_z < self.contact_threshold
        # contact_coef = in_contact.all(dim=1).float()

        # foot_xy = self.foot_pos_history[:, :, [[0, 1], [3, 4]]]
        # displacement = (
        #     (foot_xy.unsqueeze(1) - foot_xy.unsqueeze(2))
        #     .norm(dim=-1)
        #     .max(dim=1)[0]
        #     .max(dim=1)[0]
        # )

        # print(self.foot_pos_history[:, 0, [2, 5]], contact_coef * displacement)
        # foot_slide = contact_coef * displacement

        displacement = self.foot_pos_history[:, 0] - self.foot_pos_history[:, 1]
        displacement = displacement[:, [[0, 1], [3, 4]]].norm(dim=-1)

        foot_slide = displacement.mul(
            2 - 2 ** (foot_z.max(dim=1)[0] / self.contact_threshold).clamp_(0, 1)
        )

        return foot_slide

    def calc_energy_penalty(self, next_frame):
        action_energy = (
            next_frame[:, [0, 1]].pow(2).sum(1)
            + next_frame[:, 2].pow(2)
            + next_frame[:, 69:135].pow(2).mean(1)
        )
        return -0.8 * action_energy.unsqueeze(dim=1)

    def calc_action_penalty(self):
        prob_energy = self.action.abs().mean(-1, keepdim=True)
        return -0.01 * prob_energy

    def step(self, action: torch.Tensor):
        action = action * self.action_scale
        next_frame = self.get_vae_next_frame(action)
        for i in range(self.frame_skip):
            state = self.calc_env_state(next_frame[:, i])
        return state

    def calc_env_state(self, next_frame):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.is_rendered:
            self.viewer.close()

    def render(self, mode="human"):
        self.viewer.render(
            self.history[:, 0],  # 0 is the newest
            self.root_facing,
            self.root_xz,
            0.0,  # No time in this env
            self.action,
        )


class CustomRandomWalkEnv(EnvBase):
    def __init__(
        self,
        num_parallel,
        device,
        pose_vae_path,
        rendered=False,
        use_params=False,
        camera_tracking=True,
        frame_skip=1,
    ):
        super().__init__(
            num_parallel,
            device,
            pose_vae_path,
            rendered,
            use_params,
            camera_tracking,
            frame_skip,
        )

        self.max_timestep = 1000
        self.base_action = torch.zeros((self.num_parallel, self.action_dim)).to(
            self.device
        )

        self.observation_dim = (
            self.frame_dim * self.num_condition_frames + self.action_dim
        )
        high = np.inf * np.ones([self.observation_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def get_observation_components(self):
        self.base_action.normal_(0, 1)
        condition = self.get_vae_condition(normalize=False)
        return condition, self.base_action

    def reset(self, indices=None):
        self.timestep = 0
        self.substep = 0
        self.root_facing.fill_(0)
        self.root_xz.fill_(0)
        self.done.fill_(False)
        # Need to clear this if we want to use calc_foot_slide()
        self.foot_pos_history.fill_(1)

        self.reset_initial_frames()
        obs_components = self.get_observation_components()
        return torch.cat(obs_components, dim=1)

    def get_vae_next_frame(self, action):
        action = (self.base_action + action) / 2
        return super().get_vae_next_frame(action)

    def calc_env_state(self, next_frame):
        self.next_frame = next_frame
        is_external_step = self.substep == 0

        if self.substep == self.frame_skip - 1:
            self.timestep += 1
        self.substep = (self.substep + 1) % self.frame_skip

        self.integrate_root_translation(next_frame)

        self.reward.fill_(1)  # Alive bonus
        # energy_penalty = self.calc_energy_penalty(next_frame)
        # self.reward.add_(energy_penalty)
        foot_slide = self.calc_foot_slide()
        self.reward.add_(foot_slide.sum(dim=-1, keepdim=True) * -10.0)

        obs_components = self.get_observation_components()
        self.done.fill_(self.timestep >= self.max_timestep)

        self.render()

        return (
            torch.cat(obs_components, dim=1),
            self.reward,
            self.done,
            {"reset": self.timestep >= self.max_timestep},
        )

    def render(self, mode="human"):
        if self.is_rendered:
            self.viewer.render(
                self.history[:, 0],  # 0 is the newest
                self.root_facing,
                self.root_xz,
                0.0,  # No time in this env
                self.action,
            )

    def dump_additional_render_data(self):
        from common.misc_utils import POSE_CSV_HEADER

        current_frame = self.history[:, 0]
        pose_data = torch.cat(
            (current_frame[:, 0:96], current_frame[:, 189:375]), dim=-1
        )

        data_dict = {
            "pose{}.csv".format(index): {"header": POSE_CSV_HEADER}
            for index in range(pose_data.shape[0])
        }
        for index, pose in enumerate(pose_data):
            key = "pose{}.csv".format(index)
            data_dict[key]["data"] = pose.clone()

        return data_dict


class CustomTargetEnv(EnvBase):
    def __init__(
        self,
        num_parallel,
        device,
        pose_vae_path,
        rendered=False,
        use_params=False,
        camera_tracking=True,
        frame_skip=1,
    ):
        super().__init__(
            num_parallel,
            device,
            pose_vae_path,
            rendered,
            use_params,
            camera_tracking,
            frame_skip,
        )

        self.arena_length = (-60.0, 60.0)
        self.arena_width = (-40.0, 40.0)

        # 2D delta to task in root space
        target_dim = 2
        self.target = torch.zeros((self.num_parallel, target_dim)).to(self.device)

        self.observation_dim = (self.frame_dim * self.num_condition_frames) + target_dim
        high = np.inf * np.ones([self.observation_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def calc_potential(self):
        target_delta, target_angle = self.get_target_delta_and_angle()
        self.linear_potential = -target_delta.norm(dim=1).unsqueeze(1)
        self.angular_potential = target_angle.cos()

    def get_target_delta_and_angle(self):
        target_delta = self.target - self.root_xz
        target_angle = (
            torch.atan2(target_delta[:, 1], target_delta[:, 0]).unsqueeze(1)
            + self.root_facing
        )
        return target_delta, target_angle

    def get_observation_components(self):
        target_delta, _ = self.get_target_delta_and_angle()
        # Should be negative because going from global to local
        mat = self.get_rotation_matrix(-self.root_facing)
        delta = (mat * target_delta.unsqueeze(1)).sum(dim=2)
        condition = self.get_vae_condition(normalize=False)
        return condition, delta

    def reset(self, indices=None):
        if indices is None:
            self.root_facing.fill_(0)
            self.root_xz.fill_(0)
            self.reward.fill_(0)
            self.timestep = 0
            self.substep = 0
            self.done.fill_(False)
            # value bigger than contact_threshold
            self.foot_pos_history.fill_(1)

            self.reset_target()
            self.reset_initial_frames()
        else:
            self.root_facing.index_fill_(dim=0, index=indices, value=0)
            self.root_xz.index_fill_(dim=0, index=indices, value=0)
            self.reward.index_fill_(dim=0, index=indices, value=0)
            self.done.index_fill_(dim=0, index=indices, value=False)
            self.reset_target(indices)

            # value bigger than contact_threshold
            self.foot_pos_history.index_fill_(dim=0, index=indices, value=1)

        obs_components = self.get_observation_components()
        return torch.cat(obs_components, dim=1)

    def reset_target(self, indices=None, location=None):
        if location is None:
            if indices is None:
                self.target[:, 0].uniform_(*self.arena_length)
                self.target[:, 1].uniform_(*self.arena_width)
            else:
                # if indices is a pytorch tensor, this returns a new storage
                new_lengths = self.target[indices, 0].uniform_(*self.arena_length)
                self.target[:, 0].index_copy_(dim=0, index=indices, source=new_lengths)
                new_widths = self.target[indices, 1].uniform_(*self.arena_width)
                self.target[:, 1].index_copy_(dim=0, index=indices, source=new_widths)
        else:
            # Reaches this branch only with mouse click in render mode
            self.target[:, 0] = location[0]
            self.target[:, 1] = location[1]

        # l = np.random.uniform(*self.arena_length)
        # w = np.random.uniform(*self.arena_width)
        # self.target[:, 0].fill_(0)
        # self.target[:, 1].fill_(100)

        # set target to be in front
        # facing_delta = self.root_facing.clone().uniform_(-np.pi / 2, np.pi / 2)
        # angle = self.root_facing + facing_delta
        # distance = self.root_facing.clone().uniform_(20, 60)
        # self.target[:, 0].copy_((distance * angle.cos()).squeeze(1))
        # self.target[:, 1].copy_((distance * angle.sin()).squeeze(1))

        # Getting image
        # facing_delta = self.root_facing.clone().fill_(-np.pi / 6)
        # angle = self.root_facing + facing_delta
        # distance = self.root_facing.clone().fill_(40)
        # self.target[:, 0].copy_((distance * angle.cos()).squeeze(1))
        # self.target[:, 1].copy_((distance * angle.sin()).squeeze(1))

        if self.is_rendered:
            self.viewer.update_target_markers(self.target)

        # Should do this every time target is reset
        self.calc_potential()

    def calc_progress_reward(self):
        old_linear_potential = self.linear_potential
        old_angular_potential = self.angular_potential

        self.calc_potential()

        linear_progress = self.linear_potential - old_linear_potential
        angular_progress = self.angular_potential - old_angular_potential
        progress = linear_progress

        return progress

    def calc_env_state(self, next_frame):
        self.next_frame = next_frame
        is_external_step = self.substep == 0

        if self.substep == self.frame_skip - 1:
            self.timestep += 1
        self.substep = (self.substep + 1) % self.frame_skip

        self.integrate_root_translation(next_frame)

        progress = self.calc_progress_reward()

        # Check if target is reached
        # Has to be done after new potentials are calculated
        target_dist = -self.linear_potential
        target_is_close = target_dist < 2.0

        if is_external_step:
            self.reward.copy_(progress)
        else:
            self.reward.add_(progress)

        self.reward.add_(target_is_close.float() * 20.0)

        energy_penalty = self.calc_energy_penalty(next_frame)
        self.reward.add_(energy_penalty)

        # action_penalty = self.calc_action_penalty()
        # self.reward.add_(action_penalty)

        if target_is_close.any():
            reset_indices = self.parallel_ind_buf.masked_select(
                target_is_close.squeeze(1)
            )
            self.reset_target(indices=reset_indices)

        obs_components = self.get_observation_components()
        self.done.fill_(self.timestep >= self.max_timestep)

        # Everytime this function is called, should call render
        # otherwise the fps will be wrong
        self.render()

        return (
            torch.cat(obs_components, dim=1),
            self.reward,
            self.done,
            {"reset": self.timestep >= self.max_timestep},
        )

    def dump_additional_render_data(self):
        return {"extra.csv": {"header": "Target.X, Target.Z", "data": self.target[0]}}

    def render(self, mode="human"):
        if self.is_rendered:
            self.viewer.render(
                self.history[:, 0],  # 0 is the newest
                self.root_facing,
                self.root_xz,
                0.0,  # No time in this env
                self.action,
            )