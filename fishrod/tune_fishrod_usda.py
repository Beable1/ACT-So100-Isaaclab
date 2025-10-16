#!/usr/bin/env python3
"""
Complete Interactive Fishrod Physics Tuning
===========================================

Full interactive control with:
- Gravity tuning
- Individual joint stiffness and damping
- Reset to straight position
- Real-time physics updates

Usage:
    ../isaac_py.sh interactive_complete_tuning.py
"""

import sys
import os
import time
import yaml
import random
from isaacsim import SimulationApp

# Initialize simulation FIRST
print("Initializing Isaac Sim...")
simulation_app = SimulationApp({"headless": False})

# Import Omniverse modules
import omni
import omni.ui as ui
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils import extensions
from pxr import Gf, UsdPhysics, PhysxSchema, UsdGeom, UsdShade, Sdf, UsdLux

class CompleteFishrodTuner:
    """Complete interactive tuner with individual joint control"""
    
    def __init__(self, usd_stage, simulation_context, config):
        self.usd_stage = usd_stage
        self.simulation_context = simulation_context
        self.config = config
        self.physics_scene_path = config.get('physics_scene', {}).get('path', '/World/physicsScene')
        
        # Current values
        self.gravity = config.get('gravity', {}).get('initial_magnitude', 0.0)
        
        # Read joint values from USD file
        self._read_joints_from_usd()
        
        # Density values (segment density and payload density)
        self.num_segments = self._count_segments_from_usd()
        default_density = 500.0  # kg/m³ (default wood-like density)
        self.segment_density = [default_density] * self.num_segments
        self.payload_density = 1000.0  # kg/m³ (default payload density)
        
        # Original values
        self.orig_gravity = self.gravity
        self.orig_stiffness = self.base_stiffness.copy()
        self.orig_damping = self.base_damping.copy()
        self.orig_segment_density = self.segment_density.copy()
        self.orig_payload_density = self.payload_density
        
        # Test objects tracking
        self.test_objects = []
        self.last_spawn_time = 0
        self.spawn_interval = 3.0  # seconds between spawns
        self.max_objects = 10  # max test objects to keep
        
        # Create UI window
        self.create_ui()
        
        # Apply initial zero gravity to see if joints straighten
        self._apply_gravity()
    
    def _read_joints_from_usd(self):
        """Read joint stiffness and damping values from USD file"""
        from pxr import UsdPhysics
        
        # Find all revolute joints in the fishrod (excluding payload joints)
        joint_paths = []
        for prim in self.usd_stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint):
                path_str = str(prim.GetPath())
                # Only include segment joints (Joint_0_1, Joint_1_2, etc.)
                if "Joint_" in path_str and "_" in path_str.split("Joint_")[1]:
                    parts = path_str.split("Joint_")[1].split("_")
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        joint_paths.append(path_str)
        
        # Sort joints by index
        joint_paths.sort()
        self.num_joints = len(joint_paths)
        
        # Read stiffness and damping from each joint
        self.base_stiffness = []
        self.base_damping = []
        
        for joint_path in joint_paths:
            joint_prim = self.usd_stage.GetPrimAtPath(joint_path)
            
            # Read stiffness
            stiffness_attr = joint_prim.GetAttribute("drive:angular:physics:stiffness")
            if stiffness_attr and stiffness_attr.Get() is not None:
                self.base_stiffness.append(float(stiffness_attr.Get()))
            else:
                self.base_stiffness.append(1.0)  # Default
            
            # Read damping
            damping_attr = joint_prim.GetAttribute("drive:angular:physics:damping")
            if damping_attr and damping_attr.Get() is not None:
                self.base_damping.append(float(damping_attr.Get()))
            else:
                self.base_damping.append(0.5)  # Default
        
        # Current joint values (copies)
        self.joint_stiffness = self.base_stiffness.copy()
        self.joint_damping = self.base_damping.copy()
        
        print(f"✅ Read {self.num_joints} joints from USD")
        print(f"   Stiffness: {self.base_stiffness}")
        print(f"   Damping: {self.base_damping}")
    
    def _count_segments_from_usd(self):
        """Count rod segments from USD file"""
        segment_count = 0
        for prim in self.usd_stage.Traverse():
            path_str = str(prim.GetPath())
            if "/FishRod/Rod/Segment" in path_str:
                # Extract segment number
                try:
                    seg_num = int(path_str.split("Segment")[1].split("/")[0])
                    segment_count = max(segment_count, seg_num + 1)
                except (IndexError, ValueError):
                    pass
        
        print(f"✅ Found {segment_count} segments in USD")
        return segment_count
    
    def create_ui(self):
        """Create the tuning UI"""
        self.window = ui.Window("Complete Fishrod Tuner", width=450, height=900)
        
        with self.window.frame:
            with ui.VStack(spacing=10, style={"margin": 10}):
                ui.Label("🎣 Complete Fishrod Tuner", 
                        style={"font_size": 22, "color": 0xFF00AAFF})
                
                ui.Separator()
                
                # Gravity section
                with ui.CollapsableFrame("🌍 Gravity Control", height=0):
                    with ui.VStack(spacing=5):
                        with ui.HStack():
                            ui.Label("Magnitude:", width=80)
                            self.gravity_slider = ui.FloatSlider(
                                min=0.0, 
                                max=20.0,
                                height=20
                            )
                            self.gravity_slider.model.set_value(self.gravity)
                            self.gravity_slider.model.add_value_changed_fn(self._on_gravity_changed)
                        
                        self.gravity_label = ui.Label(f"Current: {self.gravity:.2f} m/s²", 
                                                      style={"color": 0xFFFFFFFF})
                
                ui.Separator()
                
                # Global scale control - DISABLED due to crashes
                with ui.CollapsableFrame("📐 Global Scale (DISABLED)", height=0, collapsed=True):
                    with ui.VStack(spacing=5):
                        ui.Label("❌ SCALING DISABLED - Causes crashes!", 
                                style={"color": 0xFFFF0000, "font_size": 12, "font_weight": "bold"})
                        ui.Label("Current: 1.0x (fixed)", 
                                style={"color": 0xFFAAAAAA, "font_size": 10})
                        
                        ui.Spacer(height=10)
                        
                        ui.Label("🔧 WORKAROUND:", 
                                style={"color": 0xFFFFAA00, "font_size": 11, "font_weight": "bold"})
                        ui.Label("1. Edit fishrod_clean.usda manually", 
                                style={"color": 0xFFFFFFFF, "font_size": 10})
                        ui.Label("2. Add scale to /FishRod/Rod:", 
                                style={"color": 0xFFFFFFFF, "font_size": 10})
                        ui.Label("   double3 xformOp:scale = (0.5, 0.5, 0.5)", 
                                style={"color": 0xFF00FF00, "font_size": 9, "font_family": "monospace"})
                        ui.Label("3. Save as fishrod_scaled.usda", 
                                style={"color": 0xFFFFFFFF, "font_size": 10})
                        ui.Label("4. Load scaled version in script", 
                                style={"color": 0xFFFFFFFF, "font_size": 10})
                        
                        ui.Spacer(height=5)
                        
                        ui.Button("📝 Show Scale Instructions", 
                                 clicked_fn=self._show_scale_instructions, 
                                 height=30)
                
                ui.Separator()
                
                # Individual joint controls
                with ui.CollapsableFrame("🔧 Individual Joint Controls", height=0, collapsed=False):
                    with ui.VStack(spacing=10):
                        self.joint_stiffness_sliders = []
                        self.joint_damping_sliders = []
                        self.joint_stiffness_labels = []
                        self.joint_damping_labels = []
                        
                        for i in range(self.num_joints):
                            # Joint header
                            ui.Label(f"Joint {i} → {i+1}", 
                                    style={"font_size": 14, "color": 0xFFFFAA00})
                            
                            # Stiffness
                            with ui.HStack():
                                ui.Label("Stiffness:", width=70)
                                stiff_slider = ui.FloatSlider(
                                    min=0.01,
                                    max=20.0,
                                    height=20
                                )
                                stiff_slider.model.set_value(self.joint_stiffness[i])
                                stiff_slider.model.add_value_changed_fn(
                                    lambda m, idx=i: self._on_joint_stiffness_changed(idx, m)
                                )
                                self.joint_stiffness_sliders.append(stiff_slider)
                            
                            stiff_label = ui.Label(f"Value: {self.joint_stiffness[i]:.2f}")
                            self.joint_stiffness_labels.append(stiff_label)
                            
                            # Damping
                            with ui.HStack():
                                ui.Label("Damping:", width=70)
                                damp_slider = ui.FloatSlider(
                                    min=0.0,
                                    max=10.0,
                                    height=20
                                )
                                damp_slider.model.set_value(self.joint_damping[i])
                                damp_slider.model.add_value_changed_fn(
                                    lambda m, idx=i: self._on_joint_damping_changed(idx, m)
                                )
                                self.joint_damping_sliders.append(damp_slider)
                            
                            damp_label = ui.Label(f"Value: {self.joint_damping[i]:.2f}")
                            self.joint_damping_labels.append(damp_label)
                            
                            if i < self.num_joints - 1:
                                ui.Spacer(height=5)
                
                ui.Separator()
                
                # Density controls
                with ui.CollapsableFrame("⚖️ Density Controls", height=0, collapsed=False):
                    with ui.VStack(spacing=10):
                        self.segment_density_sliders = []
                        self.segment_density_labels = []
                        
                        for i in range(self.num_segments):
                            # Segment header
                            ui.Label(f"Segment {i}", 
                                    style={"font_size": 14, "color": 0xFFAA00FF})
                            
                            # Density
                            with ui.HStack():
                                ui.Label("Density:", width=70)
                                density_slider = ui.FloatSlider(
                                    min=10.0,
                                    max=2000.0,
                                    height=20
                                )
                                density_slider.model.set_value(self.segment_density[i])
                                density_slider.model.add_value_changed_fn(
                                    lambda m, idx=i: self._on_segment_density_changed(idx, m)
                                )
                                self.segment_density_sliders.append(density_slider)
                            
                            density_label = ui.Label(f"Value: {self.segment_density[i]:.1f} kg/m³")
                            self.segment_density_labels.append(density_label)
                            
                            if i < self.num_segments - 1:
                                ui.Spacer(height=5)
                        
                        # Payload density
                        ui.Spacer(height=10)
                        ui.Label("Payload", 
                                style={"font_size": 14, "color": 0xFFAA00FF})
                        
                        with ui.HStack():
                            ui.Label("Density:", width=70)
                            self.payload_density_slider = ui.FloatSlider(
                                min=10.0,
                                max=2000.0,
                                height=20
                            )
                            self.payload_density_slider.model.set_value(self.payload_density)
                            self.payload_density_slider.model.add_value_changed_fn(self._on_payload_density_changed)
                        
                        self.payload_density_label = ui.Label(f"Value: {self.payload_density:.1f} kg/m³")
                        
                        ui.Spacer(height=10)
                        
                        # Summary of all densities
                        ui.Label("📊 Density Summary:", 
                                style={"font_size": 12, "color": 0xFF00AAFF, "font_weight": "bold"})
                        self.density_summary_label = ui.Label("", 
                                                              style={"font_size": 10, "color": 0xFFFFFFFF})
                        self._update_density_summary()
                
                ui.Separator()
                
                # Buttons
                with ui.HStack(spacing=10):
                    ui.Button("🔄 Reset", clicked_fn=self._on_reset, height=35)
                    ui.Button("📏 Straighten", clicked_fn=self._on_straighten, height=35)
                
                with ui.HStack(spacing=10):
                    ui.Button("💾 Export USD", clicked_fn=self._on_export, height=35)
                    ui.Button("⏸️ Pause", clicked_fn=self._on_pause, height=35)
                
                with ui.HStack(spacing=10):
                    ui.Button("🎯 Drop Ball on Rod", clicked_fn=self._on_drop_ball_on_rod, height=35)
                    ui.Button("🧊 Drop Cube on Rod", clicked_fn=self._on_drop_cube_on_rod, height=35)
                    ui.Button("🗑️ Clear All", clicked_fn=self._on_clear_objects, height=35)
                
                ui.Spacer(height=10)
                
                # Status
                self.status_label = ui.Label("Ready to tune!", 
                                            style={"color": 0xFF00FF00, "font_size": 12})
    
    def _on_gravity_changed(self, model):
        """Handle gravity slider change"""
        self.gravity = model.get_value_as_float()
        self.gravity_label.text = f"Current: {self.gravity:.2f} m/s²"
        self._apply_gravity()
    
    def _on_joint_stiffness_changed(self, joint_idx, model):
        """Handle joint stiffness change"""
        self.joint_stiffness[joint_idx] = model.get_value_as_float()
        self.joint_stiffness_labels[joint_idx].text = f"Value: {self.joint_stiffness[joint_idx]:.2f}"
        self._apply_joint_properties(joint_idx)
    
    def _on_joint_damping_changed(self, joint_idx, model):
        """Handle joint damping change"""
        self.joint_damping[joint_idx] = model.get_value_as_float()
        self.joint_damping_labels[joint_idx].text = f"Value: {self.joint_damping[joint_idx]:.2f}"
        self._apply_joint_properties(joint_idx)
    
    def _show_scale_instructions(self):
        """Show detailed scale instructions"""
        instructions = """
🔧 MANUAL SCALING INSTRUCTIONS
==============================

❌ Interactive scaling causes crashes in Isaac Sim!

✅ SAFE METHOD - Manual USD editing:

1. Open fishrod_clean.usda in a text editor
2. Find the /FishRod/Rod section
3. Add scale operation:

def Xform "Rod"
{
    double3 xformOp:scale = (0.5, 0.5, 0.5)  # 0.5x scale
    quatf xformOp:orient = (0.9914449, 0, -0.13052619, 0)
    double3 xformOp:translate = (0, 0, 0.10000000149011612)
    uniform token[] xformOpOrder = ["xformOp:scale", "xformOp:orient", "xformOp:translate"]
    
    # ... rest of segments ...
}

4. Save as fishrod_scaled_0.5x.usda
5. Modify script to load scaled version

📝 EXAMPLE SCALES:
- 0.5x = Half size
- 2.0x = Double size  
- 0.25x = Quarter size

⚠️  Keep base coordinate unchanged!
✅ Joints will scale automatically
✅ Mass calculations will update
        """
        
        print(instructions)
        self.status_label.text = "📝 Scale instructions printed to console"
        self.status_label.style = {"color": 0xFF00AAFF, "font_size": 12}
    
    def _on_segment_density_changed(self, segment_idx, model):
        """Handle segment density change"""
        self.segment_density[segment_idx] = model.get_value_as_float()
        self.segment_density_labels[segment_idx].text = f"Value: {self.segment_density[segment_idx]:.1f} kg/m³"
        self._apply_segment_density(segment_idx)
        self._update_density_summary()
    
    def _on_payload_density_changed(self, model):
        """Handle payload density change"""
        self.payload_density = model.get_value_as_float()
        self.payload_density_label.text = f"Value: {self.payload_density:.1f} kg/m³"
        self._apply_payload_density()
        self._update_density_summary()
    
    def _update_density_summary(self):
        """Update the density summary display"""
        try:
            # Create summary text
            segment_text = ", ".join([f"S{i}:{d:.0f}" for i, d in enumerate(self.segment_density)])
            summary_text = f"Segments [{segment_text}] | Payload: {self.payload_density:.0f} kg/m³"
            
            # Update label
            if hasattr(self, 'density_summary_label'):
                self.density_summary_label.text = summary_text
                
        except Exception as e:
            print(f"❌ Error updating density summary: {e}")
    
    def _apply_gravity(self):
        """Apply gravity to ALL physics scenes (Isaac Sim sometimes creates multiple)"""
        try:
            updated_scenes = []
            
            # Update ALL physics scenes to ensure no conflicts
            for prim in self.usd_stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    scene_path = prim.GetPath()
                    scene = UsdPhysics.Scene.Get(self.usd_stage, scene_path)
                    if scene:
                        scene.CreateGravityMagnitudeAttr().Set(float(self.gravity))
                        updated_scenes.append(scene_path)
            
            if updated_scenes:
                print(f"✅ Gravity set to {self.gravity:.2f} m/s² on {len(updated_scenes)} scene(s):")
                for scene_path in updated_scenes:
                    print(f"   - {scene_path}")
                self.status_label.text = f"✅ Gravity: {self.gravity:.2f} m/s² ({len(updated_scenes)} scenes)"
                self.status_label.style = {"color": 0xFF00FF00, "font_size": 12}
            else:
                print(f"❌ No physics scenes found!")
                self.status_label.text = f"❌ No physics scenes found"
                self.status_label.style = {"color": 0xFFFF0000, "font_size": 12}
                
        except Exception as e:
            print(f"❌ Gravity error: {e}")
            self.status_label.text = f"❌ Error: {e}"
            self.status_label.style = {"color": 0xFFFF0000, "font_size": 12}
    
    def _apply_joint_properties(self, joint_idx):
        """Apply stiffness and damping to specific joint"""
        try:
            joint_path = f"/FishRod/Joints/Joint_{joint_idx}_{joint_idx+1}"
            joint_prim = self.usd_stage.GetPrimAtPath(joint_path)
            
            if joint_prim.IsValid():
                # Set stiffness
                stiffness_attr = joint_prim.GetAttribute("drive:angular:physics:stiffness")
                if stiffness_attr:
                    stiffness_attr.Set(float(self.joint_stiffness[joint_idx]))
                else:
                    print(f"⚠️ No stiffness attribute found for {joint_path}")
                
                # Set damping
                damping_attr = joint_prim.GetAttribute("drive:angular:physics:damping")
                if damping_attr:
                    damping_attr.Set(float(self.joint_damping[joint_idx]))
                else:
                    print(f"⚠️ No damping attribute found for {joint_path}")
                
                print(f"✅ Joint {joint_idx}: S={self.joint_stiffness[joint_idx]:.2f}, D={self.joint_damping[joint_idx]:.2f}")
                self.status_label.text = f"✅ Updated Joint {joint_idx}"
                self.status_label.style = {"color": 0xFF00FF00, "font_size": 12}
            else:
                print(f"❌ Joint not found at {joint_path}")
                self.status_label.text = f"❌ Joint {joint_idx} not found"
                self.status_label.style = {"color": 0xFFFF0000, "font_size": 12}
        except Exception as e:
            print(f"❌ Joint {joint_idx} error: {e}")
            self.status_label.text = f"❌ Error updating joint {joint_idx}"
            self.status_label.style = {"color": 0xFFFF0000, "font_size": 12}
    
    
    def _apply_segment_density(self, segment_idx):
        """Apply density to specific segment"""
        try:
            import math
            segment_path = f"/FishRod/Rod/Segment{segment_idx}"
            segment_prim = self.usd_stage.GetPrimAtPath(segment_path)
            
            if segment_prim.IsValid():
                # Get segment dimensions
                segment_geom = UsdGeom.Cylinder(segment_prim)
                if segment_geom:
                    height = segment_geom.GetHeightAttr().Get()
                    radius = segment_geom.GetRadiusAttr().Get()
                    
                    # Calculate volume (cylinder volume = π * r² * h)
                    volume = math.pi * (radius ** 2) * height
                    
                    # Calculate mass from density
                    mass = self.segment_density[segment_idx] * volume
                    
                    # Apply mass
                    mass_attr = segment_prim.GetAttribute("physics:mass")
                    if mass_attr:
                        mass_attr.Set(float(mass))
                    else:
                        segment_prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(float(mass))
                    
                    print(f"✅ Segment {segment_idx}: Density={self.segment_density[segment_idx]:.1f} kg/m³, Mass={mass:.4f} kg")
                    self.status_label.text = f"✅ Segment {segment_idx} density updated"
                    self.status_label.style = {"color": 0xFF00FF00, "font_size": 12}
                else:
                    print(f"⚠️ Not a valid cylinder at {segment_path}")
            else:
                print(f"❌ Segment not found at {segment_path}")
                self.status_label.text = f"❌ Segment {segment_idx} not found"
                self.status_label.style = {"color": 0xFFFF0000, "font_size": 12}
        except Exception as e:
            print(f"❌ Segment {segment_idx} density error: {e}")
            self.status_label.text = f"❌ Error updating segment {segment_idx}"
            self.status_label.style = {"color": 0xFFFF0000, "font_size": 12}
    
    def _apply_payload_density(self):
        """Apply density to payload"""
        try:
            payload_path = "/FishRod/Rod/Payload"
            payload_prim = self.usd_stage.GetPrimAtPath(payload_path)
            
            if payload_prim.IsValid():
                # Get payload dimensions (cube)
                payload_geom = UsdGeom.Cube(payload_prim)
                if payload_geom:
                    size = payload_geom.GetSizeAttr().Get()
                    
                    # Calculate volume (cube volume = size³)
                    volume = size ** 3
                    
                    # Calculate mass from density
                    mass = self.payload_density * volume
                    
                    # Apply mass
                    mass_attr = payload_prim.GetAttribute("physics:mass")
                    if mass_attr:
                        mass_attr.Set(float(mass))
                    else:
                        payload_prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(float(mass))
                    
                    print(f"✅ Payload: Density={self.payload_density:.1f} kg/m³, Mass={mass:.4f} kg")
                    self.status_label.text = f"✅ Payload density updated"
                    self.status_label.style = {"color": 0xFF00FF00, "font_size": 12}
                else:
                    print(f"⚠️ Not a valid cube at {payload_path}")
            else:
                print(f"❌ Payload not found at {payload_path}")
                self.status_label.text = f"❌ Payload not found"
                self.status_label.style = {"color": 0xFFFF0000, "font_size": 12}
        except Exception as e:
            print(f"❌ Payload density error: {e}")
            self.status_label.text = f"❌ Error updating payload"
            self.status_label.style = {"color": 0xFFFF0000, "font_size": 12}
    
    def _on_straighten(self):
        """Straighten the rod by resetting joint angles"""
        try:
            # Pause simulation
            self.simulation_context.pause()
            
            # Reset joint positions to zero (straight)
            for i in range(self.num_joints):
                joint_path = f"/FishRod/Joints/Joint_{i}_{i+1}"
                joint_prim = self.usd_stage.GetPrimAtPath(joint_path)
                
                if joint_prim.IsValid():
                    # Try to get the revolute joint and set position to 0
                    from pxr import UsdPhysics
                    joint = UsdPhysics.RevoluteJoint.Get(self.usd_stage, joint_path)
                    if joint:
                        # Create drive target position at 0
                        drive_api = UsdPhysics.DriveAPI.Get(joint.GetPrim(), "angular")
                        if drive_api:
                            drive_api.CreateTargetPositionAttr().Set(0.0)
            
            # Also reset rigid body velocities
            for i in range(self.num_segments):
                segment_path = f"/FishRod/Rod/Segment{i}"
                segment_prim = self.usd_stage.GetPrimAtPath(segment_path)
                if segment_prim.IsValid():
                    rb = UsdPhysics.RigidBodyAPI(segment_prim)
                    if rb:
                        rb.CreateVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
                        rb.CreateAngularVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
            
            # Resume simulation
            self.simulation_context.play()
            
            self.status_label.text = "📏 Rod straightened"
            self.status_label.style = {"color": 0xFF00AAFF, "font_size": 12}
            print("📏 Rod straightened")
            
        except Exception as e:
            print(f"❌ Straighten error: {e}")
            self.status_label.text = f"❌ Straighten error: {e}"
            self.status_label.style = {"color": 0xFFFF0000, "font_size": 12}
    
    def _on_reset(self):
        """Reset to original values"""
        self.gravity = self.orig_gravity
        self.joint_stiffness = self.orig_stiffness.copy()
        self.joint_damping = self.orig_damping.copy()
        self.segment_density = self.orig_segment_density.copy()
        self.payload_density = self.orig_payload_density
        # Update sliders
        self.gravity_slider.model.set_value(self.gravity)
        for i in range(self.num_joints):
            self.joint_stiffness_sliders[i].model.set_value(self.joint_stiffness[i])
            self.joint_damping_sliders[i].model.set_value(self.joint_damping[i])
        for i in range(self.num_segments):
            self.segment_density_sliders[i].model.set_value(self.segment_density[i])
        self.payload_density_slider.model.set_value(self.payload_density)
        
        # Update labels
        self.gravity_label.text = f"Current: {self.gravity:.2f} m/s²"
        for i in range(self.num_joints):
            self.joint_stiffness_labels[i].text = f"Value: {self.joint_stiffness[i]:.2f}"
            self.joint_damping_labels[i].text = f"Value: {self.joint_damping[i]:.2f}"
        for i in range(self.num_segments):
            self.segment_density_labels[i].text = f"Value: {self.segment_density[i]:.1f} kg/m³"
        self.payload_density_label.text = f"Value: {self.payload_density:.1f} kg/m³"
        
        # Apply
        self._apply_gravity()
        for i in range(self.num_joints):
            self._apply_joint_properties(i)
        for i in range(self.num_segments):
            self._apply_segment_density(i)
        self._apply_payload_density()
        
        # Update density summary
        self._update_density_summary()
        
        self.status_label.text = "🔄 Reset to original values"
        self.status_label.style = {"color": 0xFF00AAFF, "font_size": 12}
        print("🔄 Reset to original values")
    
    def _on_pause(self):
        """Pause/unpause simulation"""
        if self.simulation_context.is_playing():
            self.simulation_context.pause()
            self.status_label.text = "⏸️ Simulation paused"
            print("⏸️ Paused")
        else:
            self.simulation_context.play()
            self.status_label.text = "▶️ Simulation playing"
            print("▶️ Playing")
        self.status_label.style = {"color": 0xFF00AAFF, "font_size": 12}
    
    def _on_export(self):
        """Export complete fishrod - saves ALL properties automatically (no whitelist)"""
        try:
            # Stop simulation completely during export (pause can cause issues)
            was_playing = self.simulation_context.is_playing()
            if was_playing or not self.simulation_context.is_stopped():
                print("⏹️ Stopping simulation for safe export...")
                self.simulation_context.stop()
                # Give it a moment to fully stop
                import omni
                for _ in range(3):
                    omni.kit.app.get_app().update()
            
            # Clean up temporary test objects and their materials
            print("🧹 Cleaning up temporary objects before export...")
            self._cleanup_test_objects_for_export()
            
            # === EXPORT ENTIRE STAGE ===
            # All properties are already applied in real-time through the UI
            # USD export captures EVERYTHING automatically - no need to whitelist!
            print("💾 Exporting entire stage (all fishrod properties captured automatically)...")
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"fishrod_tuned_{timestamp}.usda"
            
            self.usd_stage.Export(filename)
            print(f"   ✓ USD exported: {filename}")
            
            # === GENERATE REFERENCE CONFIG (DOCUMENTS CURRENT STATE) ===
            # Scan the actual USD stage to document what was saved
            print("📊 Scanning exported fishrod properties...")
            
            # Collect all properties from the stage
            fishrod_properties = self._collect_fishrod_properties()
            
            # Create standalone config for reference
            tuned_config = {
                'fishrod': {
                    'scale': 1.0,
                    'base_position': [0.0, 0.0, 0.0],
                    'note': 'USD file contains ALL fishrod properties - this config is for reference only'
                },
                'properties_found': fishrod_properties,
                'physics_scene': {
                    'path': '/World/physicsScene',
                    'max_position_iterations': 16,
                    'max_velocity_iterations': 4,
                    'enable_stabilization': True,
                    'bounce_threshold': 5,
                    'ccd_enabled': True,
                    'ccd_threshold': 0.01,
                    'max_velocity': 100.0,
                    'max_angular_velocity': 100.0,
                    'substeps': 4,
                    'solver_type': 'TGS'
                },
                'simulation': {
                    'max_steps': 10000,
                    'settling_steps': 60,
                    'time_delay': 0.04,
                    'render': True
                },
                'break_detection': {
                    'distance_threshold': 0.01,
                    'enabled': True
                }
            }
            
            config_filename = filename.replace('.usda', '_config.yaml')
            with open(config_filename, 'w') as f:
                yaml.dump(tuned_config, f, default_flow_style=False, sort_keys=False)
            
            self.status_label.text = f"💾 Saved! Restart script to continue"
            self.status_label.style = {"color": 0xFF00FF00, "font_size": 12}
            
            # Print comprehensive summary
            print(f"\n{'='*70}")
            print(f"💾 EXPORT COMPLETE - FULL FISHROD (ALL PROPERTIES)")
            print(f"{'='*70}")
            print(f"📄 Files created:")
            print(f"   USD:    {filename}")
            print(f"   Config: {config_filename}")
            print(f"\n📊 Properties Captured (automatically from stage):")
            if 'gravity' in fishrod_properties:
                print(f"   Gravity:          {fishrod_properties['gravity']:.2f} m/s²")
            if 'joints' in fishrod_properties:
                print(f"   Joints:           {fishrod_properties['joints']['count']}")
                print(f"     Stiffness:      {fishrod_properties['joints']['stiffness']}")
                print(f"     Damping:        {fishrod_properties['joints']['damping']}")
            if 'segments' in fishrod_properties:
                print(f"   Segments:         {fishrod_properties['segments']['count']}")
                if 'masses' in fishrod_properties['segments']:
                    print(f"     Masses:         {fishrod_properties['segments']['masses']} kg")
            if 'payload' in fishrod_properties:
                print(f"   Payload mass:     {fishrod_properties['payload']['mass']:.4f} kg")
            print(f"\n✅ USD file contains ALL fishrod properties (nothing whitelisted)")
            print(f"✅ Config file documents what was found (for reference only)")
            print(f"✅ Load {filename} directly - it's completely standalone")
            print(f"\n⚠️  IMPORTANT: Simulation stopped to prevent crashes.")
            print(f"    To continue tuning, restart the script.")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"❌ Export error: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.text = f"❌ Export error: {e}"
            self.status_label.style = {"color": 0xFFFF0000, "font_size": 12}
    
    def _collect_fishrod_properties(self):
        """Scan the stage and collect all fishrod properties (no whitelist)"""
        properties = {}
    
        # Scan for gravity
        for prim in self.usd_stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                scene = UsdPhysics.Scene.Get(self.usd_stage, prim.GetPath())
                if scene:
                    gravity_attr = scene.GetGravityMagnitudeAttr()
                    if gravity_attr:
                        properties['gravity'] = float(gravity_attr.Get())
                        break
        
        # Scan for all joints under /FishRod
        joints_info = {'count': 0, 'stiffness': [], 'damping': []}
        for prim in self.usd_stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint):
                path_str = str(prim.GetPath())
                if "/FishRod/Joints/" in path_str:
                    joints_info['count'] += 1
                    
                    stiffness_attr = prim.GetAttribute("drive:angular:physics:stiffness")
                    if stiffness_attr:
                        joints_info['stiffness'].append(float(stiffness_attr.Get()))
                    
                    damping_attr = prim.GetAttribute("drive:angular:physics:damping")
                    if damping_attr:
                        joints_info['damping'].append(float(damping_attr.Get()))
        
        if joints_info['count'] > 0:
            properties['joints'] = joints_info
        
        # Scan for all segments under /FishRod/Rod
        segments_info = {'count': 0, 'masses': []}
        for prim in self.usd_stage.Traverse():
            path_str = str(prim.GetPath())
            if "/FishRod/Rod/Segment" in path_str and prim.IsValid():
                segments_info['count'] += 1
                
                mass_attr = prim.GetAttribute("physics:mass")
                if mass_attr:
                    segments_info['masses'].append(float(mass_attr.Get()))
        
        if segments_info['count'] > 0:
            properties['segments'] = segments_info
        
        # Scan for payload
        payload_prim = self.usd_stage.GetPrimAtPath("/FishRod/Rod/Payload")
        if payload_prim.IsValid():
            mass_attr = payload_prim.GetAttribute("physics:mass")
            if mass_attr:
                properties['payload'] = {'mass': float(mass_attr.Get())}
        
        # Count total prims under /FishRod
        fishrod_prim_count = 0
        for prim in self.usd_stage.Traverse():
            if str(prim.GetPath()).startswith("/FishRod"):
                fishrod_prim_count += 1
        
        properties['total_fishrod_prims'] = fishrod_prim_count
        
        return properties
    
    def _cleanup_test_objects_for_export(self):
        """Remove all test objects and their materials before export"""
        try:
            # Remove test objects
            for obj_path in self.test_objects:
                prim = self.usd_stage.GetPrimAtPath(obj_path)
                if prim.IsValid():
                    self.usd_stage.RemovePrim(obj_path)
            
            # Remove TestObjects container if it exists
            test_objects_path = "/World/TestObjects"
            if self.usd_stage.GetPrimAtPath(test_objects_path).IsValid():
                self.usd_stage.RemovePrim(test_objects_path)
            
            # Remove test object materials (they have PhysXMaterial in the name)
            materials_path = "/World/Materials"
            materials_prim = self.usd_stage.GetPrimAtPath(materials_path)
            if materials_prim.IsValid():
                # Collect material paths to remove
                to_remove = []
                for child in materials_prim.GetChildren():
                    child_name = child.GetName()
                    # Remove materials created for test objects (Ball/Cube materials and PhysXMaterials)
                    if "Ball" in child_name or "Cube" in child_name or "PhysXMaterial" in child_name or "Rod" in child_name:
                        to_remove.append(child.GetPath())
                
                # Remove them
                for path in to_remove:
                    self.usd_stage.RemovePrim(path)
                
                print(f"   Removed {len(to_remove)} test object materials")
            
            # Clear the tracking list
            self.test_objects.clear()
            
            print(f"   ✅ Cleaned up test objects for export")
            
        except Exception as e:
            print(f"   ⚠️ Warning during cleanup: {e}")
    
    def _on_drop_ball_on_rod(self):
        """Drop a ball directly on top of the fishrod"""
        self._drop_object_on_rod("ball")
    
    def _on_drop_cube_on_rod(self):
        """Drop a cube directly on top of the fishrod"""
        self._drop_object_on_rod("cube")
    
    def _on_clear_objects(self):
        """Clear all test objects"""
        try:
            for obj_path in self.test_objects:
                prim = self.usd_stage.GetPrimAtPath(obj_path)
                if prim.IsValid():
                    self.usd_stage.RemovePrim(obj_path)
            
            self.test_objects.clear()
            self.status_label.text = f"🗑️ Cleared {len(self.test_objects)} objects"
            self.status_label.style = {"color": 0xFF00AAFF, "font_size": 12}
            print("🗑️ Cleared all test objects")
            
        except Exception as e:
            print(f"❌ Clear error: {e}")
            self.status_label.text = f"❌ Clear error: {e}"
            self.status_label.style = {"color": 0xFFFF0000, "font_size": 12}
    
    def _drop_object_on_rod(self, shape="ball"):
        """Drop an object directly on top of the fishrod"""
        try:
            # First clear all existing objects
            self._clear_all_objects()
            
            # Generate unique name
            obj_id = len(self.test_objects)
            obj_name = f"Rod{shape.capitalize()}_{obj_id}"
            obj_path = f"/World/TestObjects/{obj_name}"
            
            # Position directly above the fishrod (assuming rod is at origin)
            spawn_x = 0.0  # Center of rod
            spawn_y = 0.0  # Center of rod  
            spawn_z = 4.0  # High above rod
            
            # Create the object
            if shape == "ball":
                # Create sphere
                sphere_prim = UsdGeom.Sphere.Define(self.usd_stage, obj_path)
                sphere_prim.CreateRadiusAttr().Set(0.15)  # 15cm radius (bigger for visibility)
            else:  # cube
                # Create cube
                cube_prim = UsdGeom.Cube.Define(self.usd_stage, obj_path)
                cube_prim.CreateSizeAttr().Set(0.25)  # 25cm cube (bigger for visibility)
            
            # Set position using Xformable properly
            obj_prim = self.usd_stage.GetPrimAtPath(obj_path)
            xformable = UsdGeom.Xformable(obj_prim)
            
            # Check if translate op already exists
            translate_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            
            if translate_op is None:
                # Create new translate op
                translate_op = xformable.AddTranslateOp()
            
            # Set the position
            translate_op.Set(Gf.Vec3f(spawn_x, spawn_y, spawn_z))
            
            # Add physics with bouncing properties
            UsdPhysics.RigidBodyAPI.Apply(obj_prim)
            UsdPhysics.CollisionAPI.Apply(obj_prim)
            
            # Add PhysX material for bouncing
            physx_material_path = f"/World/Materials/{obj_name}PhysXMaterial"
            physx_material = PhysxSchema.PhysxMaterialAPI.Apply(obj_prim, physx_material_path)
            physx_material.CreateRestitutionAttr().Set(0.8)  # High bounce
            physx_material.CreateDynamicFrictionAttr().Set(0.3)  # Low friction
            physx_material.CreateStaticFrictionAttr().Set(0.3)
            
            # Add visual material
            material_path = f"/World/Materials/{obj_name}Material"
            material = UsdShade.Material.Define(self.usd_stage, material_path)
            
            # Create shader
            shader_path = f"{material_path}/Shader"
            shader = UsdShade.Shader.Define(self.usd_stage, shader_path)
            shader.CreateIdAttr("UsdPreviewSurface")
            
            # Bright colors for visibility
            colors = [
                (1.0, 0.0, 0.0),  # Red
                (0.0, 1.0, 0.0),  # Green
                (0.0, 0.0, 1.0),  # Blue
                (1.0, 1.0, 0.0),  # Yellow
                (1.0, 0.0, 1.0),  # Magenta
                (0.0, 1.0, 1.0),  # Cyan
            ]
            color = random.choice(colors)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)  # Shiny for visibility
            
            # Connect shader to material
            shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material_surface = material.CreateSurfaceOutput()
            material_surface.ConnectToSource(shader_output)
            
            # Apply material to object
            UsdShade.MaterialBindingAPI.Apply(obj_prim).Bind(material)
            
            # Track object
            self.test_objects.append(obj_path)
            
            self.status_label.text = f"🎯 Dropped {shape} on rod! (Bounce: 0.8)"
            self.status_label.style = {"color": 0xFF00FF00, "font_size": 12}
            print(f"🎯 Dropped {shape} on rod at ({spawn_x:.1f}, {spawn_y:.1f}, {spawn_z:.1f}) - Bounce: 0.8")
            
        except Exception as e:
            print(f"❌ Drop error: {e}")
            self.status_label.text = f"❌ Drop error: {e}"
            self.status_label.style = {"color": 0xFFFF0000, "font_size": 12}
    
    def _clear_all_objects(self):
        """Clear all test objects silently"""
        try:
            for obj_path in self.test_objects:
                prim = self.usd_stage.GetPrimAtPath(obj_path)
                if prim.IsValid():
                    self.usd_stage.RemovePrim(obj_path)
            self.test_objects.clear()
        except Exception as e:
            print(f"❌ Clear error: {e}")
    
    def auto_spawn_objects(self):
        """Auto-spawn objects periodically - DISABLED for manual control"""
        # Disabled auto-spawning - user controls drops manually
        pass

def create_environment(usd_stage):
    """Create environment (lighting, ground, physics scene)"""
    try:
        # Create World container
        world_path = "/World"
        if not usd_stage.GetPrimAtPath(world_path):
            world_prim = usd_stage.DefinePrim(world_path, "Xform")
        
        # Create physics scene
        physics_scene_path = "/World/physicsScene"
        if not usd_stage.GetPrimAtPath(physics_scene_path):
            physics_scene = UsdPhysics.Scene.Define(usd_stage, physics_scene_path)
            physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
            physics_scene.CreateGravityMagnitudeAttr().Set(0.0)
            
            # Add PhysX settings
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
            physx_scene.CreateBounceThresholdAttr().Set(0.2)
            physx_scene.CreateEnableStabilizationAttr().Set(True)
            physx_scene.CreateMaxPositionIterationCountAttr().Set(16)
            physx_scene.CreateMaxVelocityIterationCountAttr().Set(4)
        
        # Create lighting
        dome_light_path = "/World/DomeLight"
        if not usd_stage.GetPrimAtPath(dome_light_path):
            dome_light = UsdLux.DomeLight.Define(usd_stage, dome_light_path)
            dome_light.CreateIntensityAttr().Set(1.0)
            dome_light.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
            dome_light.CreateExposureAttr().Set(0.0)
        
        sun_light_path = "/World/SunLight"
        if not usd_stage.GetPrimAtPath(sun_light_path):
            sun_light = UsdLux.DistantLight.Define(usd_stage, sun_light_path)
            sun_light.CreateIntensityAttr().Set(3.0)
            sun_light.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.8))
            sun_light.CreateExposureAttr().Set(0.0)
            
            # Add rotation
            xformable = UsdGeom.Xformable(sun_light)
            xformable.AddRotateXYZOp().Set(Gf.Vec3f(45, 30, 0))
        
        # Create ground plane
        ground_path = "/World/GroundPlane"
        if not usd_stage.GetPrimAtPath(ground_path):
            ground_mesh = UsdGeom.Mesh.Define(usd_stage, ground_path)
            
            # Set mesh data
            ground_mesh.CreateFaceVertexCountsAttr().Set([4])
            ground_mesh.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3])
            ground_mesh.CreateNormalsAttr().Set([(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)])
            ground_mesh.CreatePointsAttr().Set([(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, 10, 0)])
            
            # Add physics
            UsdPhysics.RigidBodyAPI.Apply(ground_mesh.GetPrim())
            UsdPhysics.CollisionAPI.Apply(ground_mesh.GetPrim())
            rigid_body = UsdPhysics.RigidBodyAPI(ground_mesh.GetPrim())
            rigid_body.CreateKinematicEnabledAttr().Set(True)
            
            # Add material
            ground_mat_path = "/World/Materials/ground_mat"
            if not usd_stage.GetPrimAtPath(ground_mat_path):
                ground_material = UsdShade.Material.Define(usd_stage, ground_mat_path)
                shader = UsdShade.Shader.Define(usd_stage, f"{ground_mat_path}/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
                
                # Connect shader output to material surface
                shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                material_surface = ground_material.CreateSurfaceOutput()
                material_surface.ConnectToSource(shader_output)
            
            # Bind material
            UsdShade.MaterialBindingAPI.Apply(ground_mesh.GetPrim()).Bind(
                UsdShade.Material.Get(usd_stage, ground_mat_path)
            )
        
        print("✅ Environment created (lighting, ground, physics scene)")
        
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        raise

def load_config(config_path="config.yaml"):
    """Load configuration"""
    try:
        with open(config_path, 'r') as file:
            cfg = yaml.safe_load(file)
        print(f"✅ Config loaded")
        return cfg
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

def main():
    """Main function"""
    
    print("\n" + "="*60)
    print("🎣 COMPLETE INTERACTIVE FISHROD TUNING")
    print("="*60)
    
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    
    # Initialize simulation context
    simulation_context = SimulationContext(stage_units_in_meters=1.0)
    usd_stage = omni.usd.get_context().get_stage()
    
    # Enable extensions
    try:
        extensions.enable_extension("isaacsim.ros2.bridge")
    except:
        pass
    
    # Load clean fishrod (no environment clutter)
    fishrod_usd_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fishrod_clean.usda")
    if not os.path.isfile(fishrod_usd_path):
        print(f"❌ Clean fishrod not found at: {fishrod_usd_path}")
        simulation_app.close()
        sys.exit(1)
    
    print(f"📂 Loading clean fishrod...")
    usd_stage.GetRootLayer().subLayerPaths.append(fishrod_usd_path)
    simulation_app.update()
    
    # Create environment
    create_environment(usd_stage)
    
    # Setup physics scene
    physics_scene_path = config.get('physics_scene', {}).get('path', '/World/physicsScene')
    
    if not usd_stage.GetPrimAtPath(physics_scene_path):
        UsdPhysics.Scene.Define(usd_stage, physics_scene_path)
    
    physics_scene = UsdPhysics.Scene.Get(usd_stage, physics_scene_path)
    gravity_direction = config.get('gravity', {}).get('direction', [0.0, 0.0, -1.0])
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(*gravity_direction))
    physics_scene.CreateGravityMagnitudeAttr().Set(0.0)  # Start with zero gravity
    
    print("✅ Environment and physics scene ready (gravity = 0)")
    
    # Debug: List all physics scenes
    print("\n🔍 DEBUG: All physics scenes in stage:")
    for prim in usd_stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            scene_path = prim.GetPath()
            scene = UsdPhysics.Scene.Get(usd_stage, scene_path)
            if scene:
                gravity_attr = scene.GetGravityMagnitudeAttr()
                if gravity_attr:
                    current_gravity = gravity_attr.Get()
                    print(f"   {scene_path}: gravity = {current_gravity}")
                else:
                    print(f"   {scene_path}: no gravity attribute")
    
    # Debug: List all joints
    print("\n🔍 DEBUG: All joints in stage:")
    for prim in usd_stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint_path = prim.GetPath()
            print(f"   {joint_path}")
            # Check if it has drive attributes
            stiffness_attr = prim.GetAttribute("drive:angular:physics:stiffness")
            damping_attr = prim.GetAttribute("drive:angular:physics:damping")
            if stiffness_attr:
                print(f"     stiffness: {stiffness_attr.Get()}")
            if damping_attr:
                print(f"     damping: {damping_attr.Get()}")
    
    # Configure PhysX
    physx_scene = PhysxSchema.PhysxSceneAPI.Get(usd_stage, physics_scene_path)
    if physx_scene:
        physx_config = config.get('physics_scene', {})
        physx_scene.CreateMaxPositionIterationCountAttr().Set(physx_config.get('max_position_iterations', 16))
        physx_scene.CreateMaxVelocityIterationCountAttr().Set(physx_config.get('max_velocity_iterations', 4))
        physx_scene.CreateEnableStabilizationAttr().Set(physx_config.get('enable_stabilization', True))
        physx_scene.CreateBounceThresholdAttr().Set(physx_config.get('bounce_threshold', 5.0))
    
    print("✅ Physics scene ready (gravity = 0)")
    
    # Initialize physics
    simulation_context.initialize_physics()
    simulation_context.reset()
    simulation_app.update()
    
    # Create interactive tuner
    tuner = CompleteFishrodTuner(usd_stage, simulation_context, config)
    
    # Start simulation
    simulation_context.play()
    
    print("\n🚀 SIMULATION STARTED!")
    print("   Use the 'Complete Fishrod Tuner' window")
    print("   - Adjust gravity to see if rod bends")
    print("   - Tune individual joint stiffness/damping")
    print("   - Click 'Straighten' to reset rod position")
    print("   - Click 'Pause' to freeze and inspect")
    print("   - Click 'Export USD' to save settings")
    print("   - Click 'Drop Ball/Cube on Rod' to test gravity!")
    print("   - Each drop clears previous objects and drops fresh")
    print("="*60 + "\n")
    
    # Main simulation loop
    step = 0
    
    try:
        while simulation_app.is_running():
            simulation_context.step(render=True)
            step += 1
            
            # Auto-spawn test objects periodically
            tuner.auto_spawn_objects()
            
            # Progress
            if step % 200 == 0:
                print(f"⏱️  Step {step} - Gravity: {tuner.gravity:.1f} m/s² - Objects: {len(tuner.test_objects)}")
            
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print(f"\n\n⏸️  Interrupted at step {step}")
    
    finally:
        # Cleanup
        simulation_context.stop()
        simulation_app.close()
        
        print("\n✅ Simulation complete!")

if __name__ == "__main__":
    main()

