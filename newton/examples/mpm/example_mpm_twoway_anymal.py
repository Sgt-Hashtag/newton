# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example MPM ANYmal with 2-Way Coupling
#
# Shows ANYmal C with a pretrained policy coupled with implicit MPM sand.
# Includes two-way coupling where sand applies forces back to the robot.
#
# Example usage (via unified runner):
#   python -m newton.examples mpm_anymal --viewer gl
###########################################################################

import sys

import numpy as np
import torch
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.examples.robot.example_robot_anymal_c_walk import compute_obs, lab_to_mujoco, mujoco_to_lab
from newton.solvers import SolverImplicitMPM


@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Compute forces applied by sand to rigid bodies.

    Sum the impulses applied on each mpm grid node and convert to
    forces and torques at the body's center of mass.
    """

    i = wp.tid()

    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return

        f_world = collider_impulses[i] / dt

        X_wb = body_q[body_index]
        X_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
        wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


@wp.kernel
def subtract_body_force(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_q_res: wp.array(dtype=wp.transform),
    body_qd_res: wp.array(dtype=wp.spatial_vector),
):
    """Update the rigid bodies velocity to remove the forces applied by sand at the last step.

    This is necessary to compute the total impulses that are required to enforce the complementarity-based
    frictional contact boundary conditions.
    """

    body_id = wp.tid()

    # Remove previously applied force
    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])

    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))

    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


def _spawn_particles(builder: newton.ModelBuilder, res, bounds_lo, bounds_hi, density):
    """Spawn particles with MPM friction attribute."""
    cell_size = (bounds_hi - bounds_lo) / res
    cell_volume = np.prod(cell_size)
    radius = np.max(cell_size) * 0.5
    mass = np.prod(cell_volume) * density

    builder.add_particle_grid(
        pos=wp.vec3(bounds_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=res[0] + 1,
        dim_y=res[1] + 1,
        dim_z=res[2] + 1,
        cell_x=cell_size[0],
        cell_y=cell_size[1],
        cell_z=cell_size[2],
        mass=mass,
        jitter=2.0 * radius,
        radius_mean=radius,
        custom_attributes={"mpm:friction": 0.75},
    )


class Example:
    def __init__(
        self,
        viewer,
        voxel_size=0.05,
        particles_per_cell=3,
        tolerance=1.0e-5,
        grid_type="sparse",
    ):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.device = wp.get_device()

        # import the robot model
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        stage_path = str(asset_path / "urdf" / "anymal.urdf")
        builder.add_urdf(
            stage_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.62), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        # Disable collisions with bodies other than shanks
        for body in range(builder.body_count):
            if "SHANK" not in builder.body_key[body]:
                for shape in builder.body_shapes[body]:
                    builder.shape_flags[shape] = builder.shape_flags[shape] & ~newton.ShapeFlags.COLLIDE_PARTICLES

        builder.add_ground_plane()

        # set initial joint positions
        initial_q = {
            "RH_HAA": 0.0,
            "RH_HFE": -0.4,
            "RH_KFE": 0.8,
            "LH_HAA": 0.0,
            "LH_HFE": -0.4,
            "LH_KFE": 0.8,
            "RF_HAA": 0.0,
            "RF_HFE": 0.4,
            "RF_KFE": -0.8,
            "LF_HAA": 0.0,
            "LF_HFE": 0.4,
            "LF_KFE": -0.8,
        }
        for key, value in initial_q.items():
            builder.joint_q[builder.joint_key.index(key) + 6] = value

        for i in range(builder.joint_dof_count):
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        # ===== CRITICAL: Register MPM custom attributes BEFORE adding particles =====
        SolverImplicitMPM.register_custom_attributes(builder)

        # add sand particles
        density = 2500.0
        particle_lo = np.array([-0.5, -0.5, 0.0])
        particle_hi = np.array([0.5, 2.5, 0.15])
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )
        _spawn_particles(builder, particle_res, particle_lo, particle_hi, density)


        # density = 1500.0  # Lighter sand (sinks easier)
        
        # particle_lo = np.array([-0.5, -0.5, 0.0])
        
        # # INCREASE Z-HEIGHT: 0.15 -> 0.40 (Deep sand bed)
        # particle_hi = np.array([0.5, 2.5, 0.40])
        
        # particle_res = np.array(
        #     np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
        #     dtype=int,
        # )
        # _spawn_particles(builder, particle_res, particle_lo, particle_hi, density)

        # finalize model
        self.model = builder.finalize()

        # setup mpm solver
        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = tolerance
        mpm_options.transfer_scheme = "pic"
        mpm_options.grid_type = grid_type
        mpm_options.grid_padding = 50 if grid_type == "fixed" else 0
        mpm_options.max_active_cell_count = 1 << 15 if grid_type == "fixed" else -1
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0
        mpm_options.air_drag = 1.0

        # setup solvers
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            ls_parallel=True,
            ls_iterations=50,
            njmax=50,
        )
        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)

        # Configure collider: treat robot bodies as kinematic for two-way coupling
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
        )

        # simulation state
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Setup control policy
        self.control = self.model.control()

        q0 = wp.to_torch(self.state_0.joint_q)
        self.torch_device = q0.device
        self.joint_pos_initial = q0[7:].unsqueeze(0).detach().clone()
        self.act = torch.zeros(1, 12, device=self.torch_device, dtype=torch.float32)
        self.rearranged_act = torch.zeros(1, 12, device=self.torch_device, dtype=torch.float32)

        # Download the policy from the newton-assets repository
        policy_path = str(asset_path / "rl_policies" / "anymal_walking_policy_physx.pt")
        self.policy = torch.jit.load(policy_path, map_location=self.torch_device)

        # Pre-compute tensors that don't change during simulation
        self.lab_to_mujoco_indices = torch.tensor(
            [lab_to_mujoco[i] for i in range(len(lab_to_mujoco))], device=self.torch_device
        )
        self.mujoco_to_lab_indices = torch.tensor(
            [mujoco_to_lab[i] for i in range(len(mujoco_to_lab))], device=self.torch_device
        )
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)

        self._auto_forward = True

        # ========== TWO-WAY COUPLING SETUP ==========
        max_nodes = 1 << 20
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=self.model.device)

        # map from collider index to body index
        self.collider_body_id = self.mpm_solver.collider_body_index

        # per-body forces and torques applied by sand to robot bodies
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)

        # Initialize particle render colors
        self.particle_render_colors = wp.full(
            self.model.particle_count, value=wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=self.model.device
        )

        self.collect_collider_impulses()

        # set model on viewer and setup capture
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate_robot()
            self.graph = capture.graph

        self.sand_graph = None
        if wp.get_device().is_cuda and self.mpm_solver.grid_type == "fixed":
            with wp.ScopedCapture() as capture:
                self.simulate_sand()
            self.sand_graph = capture.graph

    def apply_control(self):
        obs = compute_obs(
            self.act,
            self.state_0,
            self.joint_pos_initial,
            self.torch_device,
            self.lab_to_mujoco_indices,
            self.gravity_vec,
            self.command,
        )
        with torch.no_grad():
            self.act = self.policy(obs)
            self.rearranged_act = torch.gather(self.act, 1, self.mujoco_to_lab_indices.unsqueeze(0))
            a = self.joint_pos_initial + 0.5 * self.rearranged_act
            a_with_zeros = torch.cat([torch.zeros(6, device=self.torch_device, dtype=torch.float32), a.squeeze(0)])
            a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
            wp.copy(self.control.joint_target_pos, a_wp)

    def collect_collider_impulses(self):
        """Collect impulses from MPM solver at contact points."""
        collider_impulses, collider_impulse_pos, collider_impulse_ids = self.mpm_solver.collect_collider_impulses(
            self.state_0
        )
        self.collider_impulse_ids.fill_(-1)
        n_colliders = min(collider_impulses.shape[0], self.collider_impulses.shape[0])
        self.collider_impulses[:n_colliders].assign(collider_impulses[:n_colliders])
        self.collider_impulse_pos[:n_colliders].assign(collider_impulse_pos[:n_colliders])
        self.collider_impulse_ids[:n_colliders].assign(collider_impulse_ids[:n_colliders])

    def simulate_robot(self):
        """Simulate robot with sand forces applied."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces from sand to robot (two-way coupling)
            wp.launch(
                compute_body_forces,
                dim=self.collider_impulse_ids.shape[0],
                inputs=[
                    self.frame_dt,
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_impulse_pos,
                    self.collider_body_id,
                    self.state_0.body_q,
                    self.model.body_com,
                    self.state_0.body_f,
                ],
            )
            # Save applied force to subtract later on
            self.body_sand_forces.assign(self.state_0.body_f)

            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, contacts=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate_sand(self):
        """Simulate sand with robot as moving boundary."""
        # Update the rigid bodies velocity to remove the forces applied by sand at the last step.
        # This prevents "double counting" momentum when the MPM solver does its collision pass.
        wp.launch(
            subtract_body_force,
            dim=self.model.body_count,
            inputs=[
                self.frame_dt,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.body_sand_forces,
                self.model.body_inv_inertia,
                self.model.body_inv_mass,
                self.state_0.body_q,
                self.state_0.body_qd,
            ],
            device=self.model.device
        )

        # sand step (in-place on frame dt)
        self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

        # Save impulses to apply back to robot in the next simulate_robot call
        self.collect_collider_impulses()

    def step(self):
        # Build command from viewer keyboard
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)

            if fwd or lat or rot:
                self._auto_forward = False

            self.command[0, 0] = float(fwd)
            self.command[0, 1] = float(lat)
            self.command[0, 2] = float(rot)

        if self._auto_forward:
            self.command[0, 0] = 1

        # compute control before graph/step
        self.apply_control()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate_robot()

        if self.sand_graph:
            wp.capture_launch(self.sand_graph)
        else:
            self.simulate_sand()

        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.1,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot went in the right direction",
            lambda q, qd: q[1] > 0.9,
        )

        forward_vel_min = wp.spatial_vector(-0.2, 0.9, -0.2, -0.8, -1.5, -0.5)
        forward_vel_max = wp.spatial_vector(0.2, 1.1, 0.2, 0.8, 1.5, 0.5)
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot is moving forward and not falling",
            lambda q, qd: newton.math.vec_inside_limits(qd, forward_vel_min, forward_vel_max),
            indices=[0],
        )
        voxel_size = self.mpm_solver.voxel_size
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "/sand",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_render_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.03)
    parser.add_argument("--particles-per-cell", "-ppc", type=float, default=3.0)
    parser.add_argument("--grid-type", "-gt", choices=["sparse", "dense", "fixed"], default="sparse")
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # This example requires a GPU device
    if wp.get_device().is_cpu:
        print("Error: This example requires a GPU device.")
        sys.exit(1)

    # Create example and load policy
    example = Example(
        viewer,
        voxel_size=args.voxel_size,
        particles_per_cell=args.particles_per_cell,
        tolerance=args.tolerance,
        grid_type=args.grid_type,
    )

    # Run via unified example runner
    newton.examples.run(example, args)