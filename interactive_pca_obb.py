#!/usr/bin/env python3
"""
Interactive OBB Comparison Demo
===============================
Compares different Oriented Bounding Box methods:
- PCA on all points (affected by point density)
- PCA on convex hull only (ignores interior points)
- Inertia tensor of convex hull (physical rotation axes)
- Minimum area OBB (mathematically optimal)

Installation:
    pip install numpy matplotlib scipy
    
    Or use the requirements file:
    pip install -r requirements_obb_demo.txt

Usage:
    python interactive_pca_obb.py

Controls:
    - Left click & drag: Move points
    - Right click: Add new point
    - Middle click: Delete point
    - Keys 1-0: Load presets
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Hull-based methods will be disabled:")
    print("  - PCA OBB (hull only)")
    print("  - Inertia OBB") 
    print("  - Min Area OBB")
    print("To enable all features: pip install scipy")

class InteractivePCAOBB:
    def __init__(self):
        # Initialize with preset 1
        self.current_preset = 1
        self.points = self.get_preset_points(self.current_preset)
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.selected_point = None
        self.dragging = False
        
        # Create scatter plot
        self.scatter = self.ax.scatter(self.points[:, 0], self.points[:, 1], 
                                      s=100, c='blue', picker=True, zorder=5)
        
        # Initialize OBB plot elements
        self.obb_line, = self.ax.plot([], [], 'r-.', linewidth=2, label='PCA OBB (all points)')
        self.hull_pca_line, = self.ax.plot([], [], 'g-.', linewidth=2, label='PCA OBB (hull only)')
        self.inertia_line, = self.ax.plot([], [], 'b-.', linewidth=2, alpha=0.8, label='Hull Inertia OBB')
        self.min_area_line, = self.ax.plot([], [], 'm--', linewidth=2, label='Min Area OBB')
        self.hull_line, = self.ax.plot([], [], 'c-', linewidth=1.5, alpha=0.5, label='Convex Hull')
        self.hull_fill = None
        self.principal_arrow = None
        self.hull_pca_arrow = None
        self.inertia_arrow = None
        
        # Info text
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                     verticalalignment='top', fontsize=10,
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Setup plot
        self.ax.set_xlim(-1, 6)
        self.ax.set_ylim(-1, 5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Disable any potential snapping
        self.ax.format_coord = lambda x, y: f'x={x:.3f}, y={y:.3f}'
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Interactive PCA OBB Demo - Drag Points to See Changes', fontsize=14)
        
        # Instructions
        instructions = ("Instructions:\n"
                       "• Click and drag points\n"
                       "• Right-click to add point\n"
                       "• Middle-click to delete point\n"
                       "• Keys 1-0: Load presets")
        self.ax.text(0.98, 0.02, instructions, transform=self.ax.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Preset info
        preset_names = {
            1: "Regular box", 2: "Box with dense edge", 3: "Ellipse (sparse)",
            4: "Ellipse (dense)", 5: "Rotated ellipse", 6: "Right triangle",
            7: "L-shape", 8: "Diagonal line", 9: "Cross shape", 0: "Random + outlier"
        }
        preset_text = f"Preset {self.current_preset}: {preset_names.get(self.current_preset, 'Custom')}"
        self.preset_text = self.ax.text(0.5, 0.02, preset_text, transform=self.ax.transAxes,
                                       horizontalalignment='center', fontsize=10,
                                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initial update
        self.update_obb()
        
    def get_preset_points(self, preset_num):
        """Get predefined point sets for testing - all centered around (2.5, 2) with similar size"""
        if preset_num == 1:
            # Regular box (slightly offset from grid)
            return np.array([
                [1.1, 1.15], [3.9, 1.15], [3.9, 2.85], [1.1, 2.85]
            ])
        elif preset_num == 2:
            # Box with dense edge
            x = np.linspace(1.2, 3.8, 10)
            bottom_edge = np.column_stack([x, np.ones_like(x) * 1.1])
            corners = np.array([[3.8, 2.9], [1.2, 2.9], [1.2, 1.1]])
            return np.vstack([bottom_edge, corners])
        elif preset_num == 3:
            # Axis-aligned ellipse (sparse)
            t = np.linspace(0, 2*np.pi, 8, endpoint=False)
            return np.column_stack([1.5*np.cos(t) + 2.5, np.sin(t) + 2])
        elif preset_num == 4:
            # Axis-aligned ellipse (dense)
            t = np.linspace(0, 2*np.pi, 30, endpoint=False)
            return np.column_stack([1.5*np.cos(t) + 2.5, np.sin(t) + 2])
        elif preset_num == 5:
            # Rotated ellipse (45 degrees)
            t = np.linspace(0, 2*np.pi, 20, endpoint=False)
            x = 1.5*np.cos(t)
            y = 0.8*np.sin(t)
            # Rotate by 45 degrees
            angle = np.pi/4
            x_rot = x*np.cos(angle) - y*np.sin(angle) + 2.5
            y_rot = x*np.sin(angle) + y*np.cos(angle) + 2
            return np.column_stack([x_rot, y_rot])
        elif preset_num == 6:
            # Right triangle (offset from grid)
            return np.array([
                [1.2, 1.1], [3.8, 1.1], [1.2, 2.9]
            ])
        elif preset_num == 7:
            # L-shape (dense)
            horizontal = np.column_stack([np.linspace(1, 3.5, 8), np.ones(8) * 1.2])
            vertical = np.column_stack([np.ones(8) * 1, np.linspace(1.2, 2.8, 8)])
            return np.vstack([horizontal, vertical[1:]])  # Avoid duplicate origin
        elif preset_num == 8:
            # Diagonal line
            t = np.linspace(0, 1, 10)
            return np.column_stack([t*2.5 + 1.2, t*1.5 + 1.2])
        elif preset_num == 9:
            # Cross/Plus shape
            horizontal = np.column_stack([np.linspace(1, 4, 7), 2*np.ones(7)])
            vertical = np.column_stack([2.5*np.ones(7), np.linspace(1, 3, 7)])
            return np.vstack([horizontal, vertical])
        elif preset_num == 10:
            # Random cluster + outlier
            np.random.seed(42)
            cluster = np.random.randn(15, 2) * 0.3 + [2.5, 2]
            outlier = np.array([[3.8, 2.9]])
            return np.vstack([cluster, outlier])
        else:
            # Default: simple box
            return np.array([[1.1, 1.1], [3.9, 1.1], [3.9, 2.9], [1.1, 2.9]])
    
    def compute_pca_obb(self, points):
        """Compute PCA-based OBB"""
        if len(points) < 2:
            return None, None, None, 0
            
        center = np.mean(points, axis=0)
        centered = points - center
        
        # Handle degenerate case
        if len(points) == 2:
            direction = centered[1] - centered[0]
            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([1, 0])
            perpendicular = np.array([-direction[1], direction[0]])
            eigenvectors = np.column_stack([direction, perpendicular])
        else:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
        
        # Project points
        projected = centered @ eigenvectors
        min_proj = np.min(projected, axis=0)
        max_proj = np.max(projected, axis=0)
        
        box_center = center + (min_proj + max_proj) / 2 @ eigenvectors.T
        extents = (max_proj - min_proj) / 2
        
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
        
        return box_center, eigenvectors, extents, angle
    
    def compute_hull_pca_obb(self, points):
        """Compute PCA OBB using only convex hull points"""
        if not HAS_SCIPY or len(points) < 3:
            return None, None, None, 0
            
        try:
            from scipy.spatial import ConvexHull
            # Get convex hull
            hull = ConvexHull(points)
            hull_pts = points[hull.vertices]
            
            # Compute PCA on hull points only
            center = np.mean(hull_pts, axis=0)
            centered = hull_pts - center
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            # Ensure right-handed coordinate system
            if np.linalg.det(eigenvectors) < 0:
                eigenvectors[:, 1] *= -1
            
            # Project ALL points (not just hull) to get full extents
            centered_all = points - center
            projected = centered_all @ eigenvectors
            min_proj = np.min(projected, axis=0)
            max_proj = np.max(projected, axis=0)
            
            box_center = center + (min_proj + max_proj) / 2 @ eigenvectors.T
            extents = (max_proj - min_proj) / 2
            
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
            
            return box_center, eigenvectors, extents, angle
            
        except:
            return None, None, None, 0
    
    def compute_inertia_obb(self, points):
        """Compute OBB using convex hull's inertia tensor"""
        if not HAS_SCIPY or len(points) < 3:
            return None, None, None, 0
            
        try:
            from scipy.spatial import ConvexHull
            
            # Get convex hull
            hull = ConvexHull(points)
            hull_pts = points[hull.vertices]
            n = len(hull_pts)
            
            # First compute area and centroid using shoelace formula
            A = 0  # area
            Cx = 0  # x-coordinate of centroid  
            Cy = 0  # y-coordinate of centroid
            
            for i in range(n):
                j = (i + 1) % n
                xi, yi = hull_pts[i]
                xj, yj = hull_pts[j]
                
                cross = xi * yj - xj * yi
                A += cross
                Cx += (xi + xj) * cross
                Cy += (yi + yj) * cross
            
            A *= 0.5
            if abs(A) < 1e-10:
                return None, None, None, 0
                
            # Compute centroid (signs cancel out properly)
            Cx = Cx / (6.0 * A)
            Cy = Cy / (6.0 * A)
            centroid = np.array([Cx, Cy])
            
            # Store sign for later use with Ixy
            sign = 1 if A > 0 else -1
            A = abs(A)
            
            # Now compute second moments about centroid
            Ixx = 0  # moment about x-axis (involves y²)
            Iyy = 0  # moment about y-axis (involves x²)
            Ixy = 0  # product of inertia
            
            for i in range(n):
                j = (i + 1) % n
                # Get vertices relative to centroid
                xi, yi = hull_pts[i] - centroid
                xj, yj = hull_pts[j] - centroid
                
                # Cross product of edge
                cross = xi * yj - xj * yi
                
                # Accumulate moments using standard formulas
                Ixx += (yi*yi + yi*yj + yj*yj) * cross
                Iyy += (xi*xi + xi*xj + xj*xj) * cross
                Ixy += (xi*yj + 2*xi*yi + 2*xj*yj + xj*yi) * cross
            
            # Apply scaling factors
            Ixx = abs(Ixx) / 12.0
            Iyy = abs(Iyy) / 12.0
            Ixy = Ixy * sign / 24.0
            
            # Build the 2x2 inertia tensor
            I = np.array([[Ixx, -Ixy], 
                         [-Ixy, Iyy]])
            
            # Get eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(I)
            
            # Sort by eigenvalue (ascending order)
            # Smaller eigenvalue = easier to rotate = major axis direction
            idx = np.argsort(eigenvalues)
            eigenvectors = eigenvectors[:, idx]
            
            # Ensure right-handed coordinate system
            if np.linalg.det(eigenvectors) < 0:
                eigenvectors[:, 1] *= -1
            
            # Project all points onto principal axes to get extents
            centered = points - centroid
            projected = centered @ eigenvectors
            min_proj = np.min(projected, axis=0)
            max_proj = np.max(projected, axis=0)
            
            # Compute final OBB parameters
            box_center = centroid + (min_proj + max_proj) / 2 @ eigenvectors.T
            extents = (max_proj - min_proj) / 2
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
            
            return box_center, eigenvectors, extents, angle
            
        except Exception as e:
            print(f"Inertia OBB error: {e}")
            return None, None, None, 0
    
    def compute_min_area_obb(self, points):
        """Compute minimum area OBB using rotating calipers"""
        if not HAS_SCIPY or len(points) < 3:
            return None, None, None, 0
            
        try:
            from scipy.spatial import ConvexHull
            from scipy.spatial.distance import cdist
            
            # Get convex hull
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            n = len(hull_points)
            
            min_area = float('inf')
            best_rect = None
            best_angle = 0
            
            # For each edge of the convex hull
            for i in range(n):
                # Get edge vector
                p1 = hull_points[i]
                p2 = hull_points[(i + 1) % n]
                edge = p2 - p1
                edge_angle = np.arctan2(edge[1], edge[0])
                
                # Create rotation matrix to align edge with x-axis
                c = np.cos(-edge_angle)
                s = np.sin(-edge_angle)
                R = np.array([[c, -s], [s, c]])
                
                # Rotate all hull points
                rotated = hull_points @ R.T
                
                # Find bounding box in rotated space
                min_x, min_y = np.min(rotated, axis=0)
                max_x, max_y = np.max(rotated, axis=0)
                
                # Calculate area
                area = (max_x - min_x) * (max_y - min_y)
                
                if area < min_area:
                    min_area = area
                    best_angle = edge_angle * 180 / np.pi
                    
                    # Store rectangle in rotated space
                    center_rotated = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
                    extents = np.array([(max_x - min_x) / 2, (max_y - min_y) / 2])
                    
                    # Rotate back to original space
                    R_inv = R.T
                    center = center_rotated @ R_inv.T
                    axes = R_inv
                    
                    best_rect = (center, axes, extents)
            
            if best_rect is None:
                return None, None, None, 0
                
            return best_rect[0], best_rect[1], best_rect[2], best_angle
            
        except Exception as e:
            return None, None, None, 0
    
    def get_obb_corners(self, center, axes, extents):
        """Get corners of OBB"""
        corners = []
        for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]:
            corner = center + axes @ (extents * [sx, sy])
            corners.append(corner)
        return np.array(corners)
    
    def update_obb(self):
        """Update OBB visualization"""
        if len(self.points) < 2:
            self.obb_line.set_data([], [])
            self.hull_pca_line.set_data([], [])
            self.inertia_line.set_data([], [])
            self.min_area_line.set_data([], [])
            self.hull_line.set_data([], [])
            if self.hull_fill:
                self.hull_fill.remove()
                self.hull_fill = None
            return
            
        # Compute PCA OBB
        center, axes, extents, angle = self.compute_pca_obb(self.points)
        
        # Compute Hull PCA OBB
        hull_center, hull_axes, hull_extents, hull_angle = self.compute_hull_pca_obb(self.points)
        
        # Compute Inertia OBB
        inertia_center, inertia_axes, inertia_extents, inertia_angle = self.compute_inertia_obb(self.points)
        
        # Compute Min Area OBB
        min_center, min_axes, min_extents, min_angle = self.compute_min_area_obb(self.points)
        
        if center is not None:
            # Update PCA OBB
            corners = self.get_obb_corners(center, axes, extents)
            self.obb_line.set_data(corners[:, 0], corners[:, 1])
            
            # Draw convex hull whenever we have enough points
            if HAS_SCIPY and len(self.points) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(self.points)
                    hull_points = self.points[hull.vertices]
                    # Close the hull by adding first point at the end
                    hull_points_closed = np.vstack([hull_points, hull_points[0]])
                    self.hull_line.set_data(hull_points_closed[:, 0], hull_points_closed[:, 1])
                    
                    # Update hull fill
                    if self.hull_fill:
                        self.hull_fill.remove()
                    self.hull_fill = Polygon(hull_points, alpha=0.1, facecolor='cyan', edgecolor='none')
                    self.ax.add_patch(self.hull_fill)
                except:
                    self.hull_line.set_data([], [])
                    if self.hull_fill:
                        self.hull_fill.remove()
                        self.hull_fill = None
            else:
                self.hull_line.set_data([], [])
                if self.hull_fill:
                    self.hull_fill.remove()
                    self.hull_fill = None
            
            # Update hull PCA OBB if available
            if hull_center is not None and hull_axes is not None and hull_extents is not None:
                hull_corners = self.get_obb_corners(hull_center, hull_axes, hull_extents)
                self.hull_pca_line.set_data(hull_corners[:, 0], hull_corners[:, 1])
                
                # Update hull PCA axis arrow
                if self.hull_pca_arrow:
                    self.hull_pca_arrow.remove()
                self.hull_pca_arrow = self.ax.arrow(hull_center[0], hull_center[1],
                                                  hull_axes[0, 0] * hull_extents[0] * 0.7,
                                                  hull_axes[1, 0] * hull_extents[0] * 0.7,
                                                  head_width=0.08, head_length=0.04,
                                                  fc='green', ec='green', alpha=0.5)
            else:
                self.hull_pca_line.set_data([], [])
                if self.hull_pca_arrow:
                    self.hull_pca_arrow.remove()
                    self.hull_pca_arrow = None
            
            # Update inertia OBB if available
            if inertia_center is not None and inertia_axes is not None and inertia_extents is not None:
                inertia_corners = self.get_obb_corners(inertia_center, inertia_axes, inertia_extents)
                self.inertia_line.set_data(inertia_corners[:, 0], inertia_corners[:, 1])
                
                # Update inertia axis arrow
                if self.inertia_arrow:
                    self.inertia_arrow.remove()
                self.inertia_arrow = self.ax.arrow(inertia_center[0], inertia_center[1],
                                                   inertia_axes[0, 0] * inertia_extents[0] * 0.6,
                                                   inertia_axes[1, 0] * inertia_extents[0] * 0.6,
                                                   head_width=0.08, head_length=0.04,
                                                   fc='magenta', ec='magenta', alpha=0.5)
            else:
                self.inertia_line.set_data([], [])
                if self.inertia_arrow:
                    self.inertia_arrow.remove()
                    self.inertia_arrow = None
            
            # Update min area OBB if available
            if min_center is not None and min_axes is not None and min_extents is not None:
                min_corners = self.get_obb_corners(min_center, min_axes, min_extents)
                self.min_area_line.set_data(min_corners[:, 0], min_corners[:, 1])
            else:
                self.min_area_line.set_data([], [])
            
            # Update principal axis arrow
            if axes is not None and extents is not None:
                if self.principal_arrow:
                    self.principal_arrow.remove()
                self.principal_arrow = self.ax.arrow(center[0], center[1],
                                                   axes[0, 0] * extents[0] * 0.8,
                                                   axes[1, 0] * extents[0] * 0.8,
                                                   head_width=0.1, head_length=0.05,
                                                   fc='red', ec='red', alpha=0.5)
            
            # Update info text
            info = f"Points: {len(self.points)}\n"
            
            if extents is not None:
                pca_area = 4 * extents[0] * extents[1]
                info += f"PCA (all) area: {pca_area:.2f}\n"
            else:
                pca_area = None
            
            if hull_extents is not None:
                hull_area = 4 * hull_extents[0] * hull_extents[1]
                info += f"PCA (hull) area: {hull_area:.2f}\n"
            
            if inertia_extents is not None:
                inertia_area = 4 * inertia_extents[0] * inertia_extents[1]
                info += f"Inertia area: {inertia_area:.2f}\n"
            
            if min_extents is not None:
                min_area = 4 * min_extents[0] * min_extents[1]
                info += f"Min area: {min_area:.2f}\n"
                if pca_area is not None:
                    info += f"PCA vs Min: +{((pca_area/min_area - 1)*100):.1f}%"
            
            self.info_text.set_text(info)
        
        # Update legend
        self.ax.legend(loc='upper right')
        self.fig.canvas.draw_idle()
    
    def on_press(self, event):
        """Handle mouse press"""
        if event.inaxes != self.ax:
            return
            
        if event.button == 1:  # Left click - select point
            # Find closest point
            distances = np.sqrt((self.points[:, 0] - event.xdata)**2 + 
                              (self.points[:, 1] - event.ydata)**2)
            if np.min(distances) < 0.2:  # Within threshold
                self.selected_point = np.argmin(distances)
                self.dragging = True
                
        elif event.button == 3:  # Right click - add point
            # Use exact coordinates without any rounding
            new_point = np.array([[float(event.xdata), float(event.ydata)]])
            self.points = np.vstack([self.points, new_point])
            self.scatter.set_offsets(self.points)
            self.current_preset = None  # Custom points
            self.preset_text.set_text("Custom configuration")
            self.update_obb()
            
        elif event.button == 2:  # Middle click - delete point
            if len(self.points) > 2:
                distances = np.sqrt((self.points[:, 0] - event.xdata)**2 + 
                                  (self.points[:, 1] - event.ydata)**2)
                if np.min(distances) < 0.2:
                    idx = np.argmin(distances)
                    self.points = np.delete(self.points, idx, axis=0)
                    self.scatter.set_offsets(self.points)
                    self.current_preset = None  # Custom points
                    self.preset_text.set_text("Custom configuration")
                    self.update_obb()
    
    def on_motion(self, event):
        """Handle mouse motion"""
        if self.dragging and event.inaxes == self.ax:
            # Use exact float coordinates
            self.points[self.selected_point] = [float(event.xdata), float(event.ydata)]
            self.scatter.set_offsets(self.points)
            if self.current_preset is not None:
                self.current_preset = None  # Custom points
                self.preset_text.set_text("Custom configuration")
            self.update_obb()
    
    def on_key_press(self, event):
        """Handle keyboard input for presets"""
        if event.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            # Load preset
            preset_num = int(event.key) if event.key != '0' else 10
            self.current_preset = preset_num
            self.points = self.get_preset_points(preset_num)
            self.scatter.set_offsets(self.points)
            
            # Update preset text
            preset_names = {
                1: "Regular box", 2: "Box with dense edge", 3: "Ellipse (sparse)",
                4: "Ellipse (dense)", 5: "Rotated ellipse", 6: "Right triangle",
                7: "L-shape", 8: "Diagonal line", 9: "Cross shape", 10: "Random + outlier"
            }
            preset_text = f"Preset {preset_num}: {preset_names.get(preset_num, 'Custom')}"
            self.preset_text.set_text(preset_text)
            
            self.update_obb()
    
    def on_release(self, event):
        """Handle mouse release"""
        self.dragging = False
        self.selected_point = None

# Create and show the interactive app
app = InteractivePCAOBB()
plt.show()

print("\n🎮 Interactive OBB Comparison Demo")
print("="*50)
print("Legend:")
print("• Red dash-dot = PCA OBB (all points)")
print("• Green dotted = PCA OBB (hull points only)")
print("• Magenta solid = Inertia OBB (physical rotation axes)")
print("• Blue dashed = Minimum Area OBB (optimal)")
print("• Cyan = Convex hull")

print("\nPresets (press keys 1-0):")
print("1. Regular box - All methods should match")
print("2. Box with dense edge - Shows PCA bias")
print("3. Sparse ellipse - PCA may not align with axes")
print("4. Dense ellipse - Better PCA alignment")
print("5. Rotated ellipse - Tests diagonal alignment")
print("6. Right triangle - Asymmetric shape")
print("7. L-shape - Non-convex example")
print("8. Diagonal line - Min area perfect, PCA off")
print("9. Cross shape - Complex symmetric shape")
print("0. Random cluster + outlier - Shows outlier effects")

print("\nExperiments:")
print("• Regular shapes: All methods similar (inertia = physical axes)")
print("• L-shape: Inertia follows mass distribution")
print("• Diagonal line: Min area perfect, inertia may differ")
print("• Dense edge: Only all-points PCA affected")
print("• Interior points: Only affect all-points PCA")
print("\nInertia OBB shows how the shape would naturally rotate!")
