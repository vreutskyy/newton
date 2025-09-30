import numpy as np

def compute_rectangle_inertia(width, height, mass=1.0):
    """Compute inertia tensor of a rectangle about its centroid"""
    Ixx = mass * height**2 / 12
    Iyy = mass * width**2 / 12
    Ixy = 0  # Rectangle aligned with axes has no cross term
    return np.array([[Ixx, -Ixy], [-Ixy, Iyy]])

def compute_polygon_inertia(vertices, density=1.0):
    """Compute inertia tensor of a polygon using Green's theorem"""
    n = len(vertices)
    
    # Compute centroid and area
    cx = cy = area = 0
    for i in range(n):
        j = (i + 1) % n
        v = vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1]
        area += v
        cx += (vertices[i][0] + vertices[j][0]) * v
        cy += (vertices[i][1] + vertices[j][1]) * v
    area *= 0.5
    
    if abs(area) < 1e-10:
        return np.zeros((2, 2)), 0, np.zeros(2)
        
    centroid = np.array([cx/(3*area), cy/(3*area)])
    mass = density * abs(area)
    
    # Compute inertia tensor about centroid
    Ixx = Iyy = Ixy = 0
    for i in range(n):
        p0 = vertices[i] - centroid
        p1 = vertices[(i+1)%n] - centroid
        a = p0[0]*p1[1] - p1[0]*p0[1]
        Ixx += (p0[1]**2 + p0[1]*p1[1] + p1[1]**2) * a
        Iyy += (p0[0]**2 + p0[0]*p1[0] + p1[0]**2) * a
        Ixy += (p0[0]*p1[1] + 2*p0[0]*p0[1] + 2*p1[0]*p1[1] + p1[0]*p0[1]) * a
    
    Ixx = abs(Ixx) / 12
    Iyy = abs(Iyy) / 12
    Ixy = abs(Ixy) / 24
    
    I_centroid = np.array([[Ixx, -Ixy], [-Ixy, Iyy]])
    
    return I_centroid, mass, centroid

def test_rectangle_decomposition():
    """Test if two triangles give same inertia as full rectangle"""
    # Rectangle: width=4, height=2, centered at origin
    width = 4.0
    height = 2.0
    
    # Rectangle vertices
    v0 = np.array([-width/2, -height/2])  # bottom-left
    v1 = np.array([width/2, -height/2])   # bottom-right
    v2 = np.array([width/2, height/2])    # top-right
    v3 = np.array([-width/2, height/2])   # top-left
    
    # Method 1: Direct rectangle formula
    rect_mass = width * height  # density = 1
    I_rect_formula = compute_rectangle_inertia(width, height, rect_mass)
    print("Rectangle Inertia (direct formula):")
    print(f"  Ixx = {I_rect_formula[0,0]:.6f}")
    print(f"  Iyy = {I_rect_formula[1,1]:.6f}")
    print(f"  Ixy = {I_rect_formula[0,1]:.6f}")
    print(f"  Mass = {rect_mass:.6f}")
    
    # Also compute rectangle using polygon formula for verification
    rect_vertices = [v0, v1, v2, v3]
    I_rect, mass_poly, centroid_poly = compute_polygon_inertia(rect_vertices)
    print("\nRectangle Inertia (polygon formula):")
    print(f"  Ixx = {I_rect[0,0]:.6f}")
    print(f"  Iyy = {I_rect[1,1]:.6f}")
    print(f"  Ixy = {I_rect[0,1]:.6f}")
    print(f"  Centroid = [{centroid_poly[0]:.6f}, {centroid_poly[1]:.6f}]")
    
    # Method 2: Sum of two triangles
    # Triangle 1: v0, v1, v2
    I1, m1, c1 = compute_polygon_inertia([v0, v1, v2])
    # Triangle 2: v0, v2, v3
    I2, m2, c2 = compute_polygon_inertia([v0, v2, v3])
    
    print(f"\nTriangle 1 vertices: v0={v0}, v1={v1}, v2={v2}")
    print(f"Triangle 1 (v0, v1, v2):")
    print(f"  Mass = {m1:.6f}")
    print(f"  Centroid = [{c1[0]:.6f}, {c1[1]:.6f}]")
    print(f"  I about centroid = [[{I1[0,0]:.3f}, {I1[0,1]:.3f}], [{I1[1,0]:.3f}, {I1[1,1]:.3f}]]")
    
    print(f"\nTriangle 2 vertices: v0={v0}, v2={v2}, v3={v3}")
    print(f"Triangle 2 (v0, v2, v3):")
    print(f"  Mass = {m2:.6f}")
    print(f"  Centroid = [{c2[0]:.6f}, {c2[1]:.6f}]")
    print(f"  I about centroid = [[{I2[0,0]:.3f}, {I2[0,1]:.3f}], [{I2[1,0]:.3f}, {I2[1,1]:.3f}]]")
    
    # Total mass
    total_mass = m1 + m2
    
    # Combined centroid (should be origin for symmetric rectangle)
    combined_centroid = (m1 * c1 + m2 * c2) / total_mass
    
    # Use parallel axis theorem to shift each triangle's inertia to combined centroid
    # I_new = I_old + m * (r^2 * I - r * r^T)
    r1 = c1 - combined_centroid
    print(f"\n  r1 (offset from tri1 to combined) = [{r1[0]:.3f}, {r1[1]:.3f}]")
    parallel_term1 = m1 * (np.dot(r1, r1) * np.eye(2) - np.outer(r1, r1))
    print(f"  Parallel axis term1 = [[{parallel_term1[0,0]:.3f}, {parallel_term1[0,1]:.3f}], [{parallel_term1[1,0]:.3f}, {parallel_term1[1,1]:.3f}]]")
    I1_shifted = I1 + parallel_term1
    
    r2 = c2 - combined_centroid
    print(f"\n  r2 (offset from tri2 to combined) = [{r2[0]:.3f}, {r2[1]:.3f}]")
    parallel_term2 = m2 * (np.dot(r2, r2) * np.eye(2) - np.outer(r2, r2))
    print(f"  Parallel axis term2 = [[{parallel_term2[0,0]:.3f}, {parallel_term2[0,1]:.3f}], [{parallel_term2[1,0]:.3f}, {parallel_term2[1,1]:.3f}]]")
    I2_shifted = I2 + parallel_term2
    
    # Sum the shifted inertias
    I_total_centroid = I1_shifted + I2_shifted
    
    print("\nTwo Triangles Inertia (sum):")
    print(f"  Ixx = {I_total_centroid[0,0]:.6f}")
    print(f"  Iyy = {I_total_centroid[1,1]:.6f}")
    print(f"  Ixy = {I_total_centroid[0,1]:.6f}")
    print(f"  Mass = {total_mass:.6f}")
    print(f"  Combined centroid = [{combined_centroid[0]:.6f}, {combined_centroid[1]:.6f}]")
    
    # Compare
    diff = np.abs(I_rect - I_total_centroid)
    print("\nDifference:")
    print(f"  ΔIxx = {diff[0,0]:.6e}")
    print(f"  ΔIyy = {diff[1,1]:.6e}")
    print(f"  ΔIxy = {diff[0,1]:.6e}")
    
    # Test with different decomposition
    print("\n" + "="*50)
    print("Alternative decomposition (diagonal split):")
    
    # Triangle 1: v0, v1, v3
    I1_alt, m1_alt, c1_alt = compute_polygon_inertia([v0, v1, v3])
    # Triangle 2: v1, v2, v3
    I2_alt, m2_alt, c2_alt = compute_polygon_inertia([v1, v2, v3])
    
    # Total mass
    total_mass_alt = m1_alt + m2_alt
    
    # Combined centroid
    combined_centroid_alt = (m1_alt * c1_alt + m2_alt * c2_alt) / total_mass_alt
    
    # Use parallel axis theorem
    r1_alt = c1_alt - combined_centroid_alt
    I1_shifted_alt = I1_alt + m1_alt * (np.dot(r1_alt, r1_alt) * np.eye(2) - np.outer(r1_alt, r1_alt))
    
    r2_alt = c2_alt - combined_centroid_alt
    I2_shifted_alt = I2_alt + m2_alt * (np.dot(r2_alt, r2_alt) * np.eye(2) - np.outer(r2_alt, r2_alt))
    
    I_total_centroid_alt = I1_shifted_alt + I2_shifted_alt
    
    print(f"  Ixx = {I_total_centroid_alt[0,0]:.6f}")
    print(f"  Iyy = {I_total_centroid_alt[1,1]:.6f}")
    print(f"  Ixy = {I_total_centroid_alt[0,1]:.6f}")
    print(f"  Mass = {total_mass_alt:.6f}")
    
    diff_alt = np.abs(I_rect - I_total_centroid_alt)
    print("\nDifference from rectangle:")
    print(f"  ΔIxx = {diff_alt[0,0]:.6e}")
    print(f"  ΔIyy = {diff_alt[1,1]:.6e}")
    print(f"  ΔIxy = {diff_alt[0,1]:.6e}")

if __name__ == "__main__":
    test_rectangle_decomposition()
