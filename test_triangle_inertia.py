import numpy as np

def compute_triangle_centroid_simple(p0, p1, p2):
    """Simple centroid calculation - average of vertices"""
    return (p0 + p1 + p2) / 3.0

def compute_triangle_inertia_simple(p0, p1, p2, density=1.0):
    """Compute inertia of a triangle using standard formulas"""
    # Triangle area
    area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
    mass = density * area
    
    # Centroid (simple average)
    centroid = (p0 + p1 + p2) / 3.0
    
    # Translate vertices to centroid
    v0 = p0 - centroid
    v1 = p1 - centroid  
    v2 = p2 - centroid
    
    # For a triangle about its centroid:
    # Ixx = (m/12) * (y0^2 + y1^2 + y2^2 + y0*y1 + y0*y2 + y1*y2)
    # Iyy = (m/12) * (x0^2 + x1^2 + x2^2 + x0*x1 + x0*x2 + x1*x2)
    # Ixy = (m/12) * (x0*y0 + x1*y1 + x2*y2 + 0.5*(x0*y1 + x1*y0 + x0*y2 + x2*y0 + x1*y2 + x2*y1))
    
    Ixx = mass / 12 * (v0[1]**2 + v1[1]**2 + v2[1]**2 + v0[1]*v1[1] + v0[1]*v2[1] + v1[1]*v2[1])
    Iyy = mass / 12 * (v0[0]**2 + v1[0]**2 + v2[0]**2 + v0[0]*v1[0] + v0[0]*v2[0] + v1[0]*v2[0])
    Ixy = mass / 12 * (v0[0]*v0[1] + v1[0]*v1[1] + v2[0]*v2[1] + 
                       0.5*(v0[0]*v1[1] + v1[0]*v0[1] + v0[0]*v2[1] + 
                            v2[0]*v0[1] + v1[0]*v2[1] + v2[0]*v1[1]))
    
    return np.array([[Ixx, -Ixy], [-Ixy, Iyy]]), mass, centroid

def test_simple_rectangle():
    """Test a simple 2x1 rectangle centered at origin"""
    # Rectangle vertices
    v0 = np.array([-1, -0.5])
    v1 = np.array([1, -0.5])
    v2 = np.array([1, 0.5])
    v3 = np.array([-1, 0.5])
    
    # Rectangle properties
    width = 2.0
    height = 1.0
    rect_area = width * height
    rect_mass = rect_area  # density = 1
    
    # Direct formula for rectangle about its centroid
    Ixx_rect = rect_mass * height**2 / 12
    Iyy_rect = rect_mass * width**2 / 12
    
    print("Rectangle (2x1):")
    print(f"  Direct formula: Ixx = {Ixx_rect:.4f}, Iyy = {Iyy_rect:.4f}")
    
    # Split into two triangles: (v0,v1,v2) and (v0,v2,v3)
    I1, m1, c1 = compute_triangle_inertia_simple(v0, v1, v2)
    I2, m2, c2 = compute_triangle_inertia_simple(v0, v2, v3)
    
    print(f"\nTriangle 1: mass = {m1:.4f}, centroid = [{c1[0]:.4f}, {c1[1]:.4f}]")
    print(f"  I = [[{I1[0,0]:.4f}, {I1[0,1]:.4f}], [{I1[1,0]:.4f}, {I1[1,1]:.4f}]]")
    
    print(f"\nTriangle 2: mass = {m2:.4f}, centroid = [{c2[0]:.4f}, {c2[1]:.4f}]")
    print(f"  I = [[{I2[0,0]:.4f}, {I2[0,1]:.4f}], [{I2[1,0]:.4f}, {I2[1,1]:.4f}]]")
    
    # Combined centroid
    total_mass = m1 + m2
    combined_centroid = (m1 * c1 + m2 * c2) / total_mass
    
    print(f"\nCombined: mass = {total_mass:.4f}, centroid = [{combined_centroid[0]:.4f}, {combined_centroid[1]:.4f}]")
    
    # Shift both triangles to combined centroid using parallel axis theorem
    r1 = c1 - combined_centroid
    I1_shifted = I1 + m1 * (np.dot(r1, r1) * np.eye(2) - np.outer(r1, r1))
    
    r2 = c2 - combined_centroid
    I2_shifted = I2 + m2 * (np.dot(r2, r2) * np.eye(2) - np.outer(r2, r2))
    
    # Sum
    I_total = I1_shifted + I2_shifted
    
    print(f"\nSum of triangles: Ixx = {I_total[0,0]:.4f}, Iyy = {I_total[1,1]:.4f}, Ixy = {I_total[0,1]:.4f}")
    print(f"Difference: ΔIxx = {abs(I_total[0,0] - Ixx_rect):.6f}, ΔIyy = {abs(I_total[1,1] - Iyy_rect):.6f}")

if __name__ == "__main__":
    print("Starting test...")
    test_simple_rectangle()
    print("Test complete.")
