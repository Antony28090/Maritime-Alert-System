import numpy as np
from shapely.geometry import Point, LineString

def distance_from_polyline(point, points):
    """
    Calculates minimum distance (in km) from a point [lat, lon] to a polyline.
    Returns (distance_km, closest_segment_index)
    """
    min_dist = float('inf')
    closest_seg_idx = -1
    p = Point(point[1], point[0]) # Shapely uses (lon, lat)
    
    for i in range(len(points) - 1):
        p1 = (points[i][1], points[i][0])
        p2 = (points[i+1][1], points[i+1][0])
        
        line = LineString([p1, p2])
        dist = line.distance(p) * 111.32 # Degrees to km
        if dist < min_dist:
            min_dist = dist
            closest_seg_idx = i
            
    return min_dist, closest_seg_idx

def is_sri_lankan_side(point, points):
    """
    Determines if a point [lat, lon] is on the Sri Lankan side (East) of the IMBL.
    Assumes IMBL_POINTS are ordered generally North to South.
    """
    # 1. Find closest segment
    _, idx = distance_from_polyline(point, points)
    if idx == -1: return False
    
    # 2. Check side of that segment using Cross Product
    # Vector AB: Segment
    # Vector AP: Point relative to Start
    
    A = points[idx]
    B = points[idx+1]
    P = point
    
    # Coordinates: [Lat, Lon] -> [y, x]
    # Cross Product (2D) = (Bx - Ax)*(Py - Ay) - (By - Ay)*(Px - Ax)
    # x is Lon, y is Lat
    
    ax, ay = A[1], A[0]
    bx, by = B[1], B[0]
    px, py = P[1], P[0]
    
    # Vector AB
    ab_x = bx - ax
    ab_y = by - ay
    
    # Vector AP
    ap_x = px - ax
    ap_y = py - ay
    
    # Cross product z-component
    cross_product = (ab_x * ap_y) - (ab_y * ap_x)
    
    # Interpretation:
    # If line is North -> South (Lat decreasing):
    # Vector AB points "Down".
    # Sri Lanka (East) is to the "Left" of the vector?
    # Let's visualize: 
    # A=(10, 80), B=(9, 79). AB vector = (-1 lon, -1 lat).
    # Point P=(9.5, 80) [East, SL]. AP vector = (0 lon, -0.5 lat).
    # Wait. A=(Lat 10, Lon 80), B=(Lat 9, Lon 79).
    # A=[80, 10], B=[79, 9].
    # AB = [-1, -1].
    # P=[80.5, 9.5]. AP = [0.5, -0.5].
    # CP = (-1 * -0.5) - (-1 * 0.5) = 0.5 - (-0.5) = 1.0. Positive.
    # Point Q=[78.5, 9.5] [West, India]. AQ = [-1.5, -0.5].
    # CP = (-1 * -0.5) - (-1 * -1.5) = 0.5 - 1.5 = -1.0. Negative.
    
    # So Positive Cross Product means "Left" relative to AB (which is East here).
    # Therefore, if CP > 0, it is Sri Lankan side.
    
    return cross_product > 0
