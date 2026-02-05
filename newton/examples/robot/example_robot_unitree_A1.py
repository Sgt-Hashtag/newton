import mujoco
import warp as wp
import os
import numpy as np

import newton
import newton.examples
import newton.utils

class Example:
    def __init__(self, viewer, num_worlds=1, args=None):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds
        self.viewer = viewer
        self.device = wp.get_device()

        articulation_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(articulation_builder)
        
        # 1. Physics Tuning
        articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e4, limit_kd=1.0e2, friction=0.01
        )
        articulation_builder.default_shape_cfg.ke = 1.0e5
        articulation_builder.default_shape_cfg.kd = 1.0e3

        # 2. Asset Loading
        asset_file = "/home/isaac/Documents/newton/newton/examples/assets/A1/a1.usd"
        articulation_builder.add_usd(
            asset_file,
            collapse_fixed_joints=False,
            enable_self_collisions=True,
        )



        # 3. STANDING POSE FIX
        # Try this inverted pose: Hip=0, Thigh=-0.6, Calf=1.2
        self.base_standing_pose = [0.0, -0.6, 1.2] * 4 
        
        articulation_builder.joint_q[0:3] = [0.0, 0.0, 0.42]
        articulation_builder.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quaternion

        num_dofs = articulation_builder.joint_dof_count
        if len(articulation_builder.joint_q) >= 7 + num_dofs:
            for i in range(num_dofs):
                articulation_builder.joint_q[7 + i] = self.base_standing_pose[i]

        # 4. HIGH STIFFNESS
        for i in range(num_dofs):
            if i % 3 == 0:          # hip
                articulation_builder.joint_target_ke[i] = 400.0
                articulation_builder.joint_target_kd[i] = 40.0
            elif i % 3 == 1:        # thigh
                articulation_builder.joint_target_ke[i] = 1200.0
                articulation_builder.joint_target_kd[i] = 80.0
            else:                   # calf
                articulation_builder.joint_target_ke[i] = 800.0
                articulation_builder.joint_target_kd[i] = 60.0


        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.num_worlds):
            builder.add_world(articulation_builder)
        builder.add_ground_plane()

        self.model = builder.finalize()

        print("\n=== JOINT ORDER ===")
        for i, name in enumerate(self.model.joint_axis):
            print(i, name)


        # 5. SOLVER FIX (Crucial for the 'njmax' error in your screenshot)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            cone=mujoco.mjtCone.mjCONE_ELLIPTIC,
            njmax=200,    # Increased from your error log's requirement
            nconmax=100,  # Increased to prevent overflow
            iterations=100,
            use_mujoco_contacts=args.use_mujoco_contacts if args else False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        target_array = np.array(self.base_standing_pose[:num_dofs], dtype=np.float32)
        self.control.joint_target = wp.array(target_array, device=self.device)


        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        
        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        # Update target in case you want to animate it later
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        for _ in range(self.sim_substeps):
            self.contacts = self.model.collide(
                self.state_0, collision_pipeline=self.collision_pipeline
            )
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0


    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=1)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args.num_worlds, args)
    newton.examples.run(example, args)