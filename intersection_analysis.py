import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.ops import unary_union
from scipy.spatial.distance import cdist

class PathAnalyzer:
    def __init__(self, tolerance=1e-6, smooth_window=5):
        """
        Initialize the path analyzer.
        
        Parameters:
        tolerance (float): Minimum distance to consider points distinct
        smooth_window (int): Window size for smoothing (must be odd)
        """
        self.tolerance = tolerance
        self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        
    def smooth_path(self, x_coords, y_coords, method='savgol', window=None):
        """
        Smooth the path using various methods.
        
        Parameters:
        x_coords (array-like): X coordinates
        y_coords (array-like): Y coordinates
        method (str): 'savgol' for Savitzky-Golay or 'moving_avg' for moving average
        window (int): Window size for smoothing (must be odd), overrides smooth_window if provided
        
        Returns:
        tuple: (smoothed_x, smoothed_y)
        """
        from scipy.signal import savgol_filter
        
        window = window if window is not None else self.smooth_window
        window = window if window % 2 == 1 else window + 1
        
        if method == 'savgol':
            # Savitzky-Golay filter: polynomial fitting
            # Order 3 polynomial by default
            x_smooth = savgol_filter(x_coords, window, 3)
            y_smooth = savgol_filter(y_coords, window, 3)
            
        elif method == 'moving_avg':
            # Simple moving average
            kernel = np.ones(window) / window
            x_smooth = np.convolve(x_coords, kernel, mode='same')
            y_smooth = np.convolve(y_coords, kernel, mode='same')
            
            # Fix endpoints
            half = window // 2
            x_smooth[:half] = x_coords[:half]
            x_smooth[-half:] = x_coords[-half:]
            y_smooth[:half] = y_coords[:half]
            y_smooth[-half:] = y_coords[-half:]
            
        else:
            raise ValueError("Method must be 'savgol' or 'moving_avg'")
    
    def _create_segments(self, path):
        """Convert path into line segments."""
        return [LineString([path[i], path[i+1]]) 
                for i in range(len(path)-1)]
    
    def find_intersections(self, x_coords, y_coords, min_segment_distance=200):
        """
        Find all self-intersection points in a path.
        
        Parameters:
        x_coords (array-like): X coordinates of the path
        y_coords (array-like): Y coordinates of the path
        min_segment_distance (int): Minimum number of indices that must separate segments
                                  for an intersection to be counted
        
        Returns:
        tuple: (intersection_points, intersection_angles, intersection_density)
        """
        path = list(zip(x_coords, y_coords))
        segments = self._create_segments(path)
        
        # Find all intersections
        intersection_points = []
        intersection_angles = []
        
        for i, seg1 in enumerate(segments):
            # Only check segments that are sufficiently far apart
            for j, seg2 in enumerate(segments[i+min_segment_distance:], i+min_segment_distance):
                if seg1.intersects(seg2):
                    # Get intersection point
                    intersection = seg1.intersection(seg2)
                    if intersection.geom_type == 'Point':
                        point = (intersection.x, intersection.y)
                        
                        # Calculate angle between segments
                        vec1 = np.array(path[i+1]) - np.array(path[i])
                        vec2 = np.array(path[j+1]) - np.array(path[j])
                        angle = np.abs(np.degrees(
                            np.arctan2(np.cross(vec1, vec2), np.dot(vec1, vec2))
                        ))
                        
                        intersection_points.append(point)
                        intersection_angles.append(angle)
        
        # Calculate path length
        total_path_length = sum(seg.length for seg in segments)
        
        # Calculate intersection density
        intersection_density = len(intersection_points) / total_path_length if total_path_length > 0 else 0
        
        return intersection_points, intersection_angles, intersection_density
    
    def analyze_and_plot(self, x_coords, y_coords, title="Path Analysis", min_segment_distance=200):
        """
        Analyze path and create visualization.
        
        Parameters:
        x_coords (array-like): X coordinates of the path
        y_coords (array-like): Y coordinates of the path
        title (str): Plot title
        
        Returns:
        dict: Analysis results
        """
        intersections, angles, density = self.find_intersections(x_coords, y_coords, min_segment_distance)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Path plot
        plt.subplot(121)
        plt.plot(x_coords, y_coords, 'b-', alpha=0.6, label='Path')
        if intersections:
            intersection_x, intersection_y = zip(*intersections)
            plt.scatter(intersection_x, intersection_y, c='r', s=50, 
                       alpha=0.6, label='Intersections')
        plt.title(f'{title}\nIntersection Density: {density:.3f}')
        plt.legend()
        plt.axis('equal')
        
        # Angle histogram
        if angles:
            plt.subplot(122)
            plt.hist(angles, bins=18, range=(0, 180), edgecolor='black')
            plt.title('Intersection Angles Distribution')
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Count')
        
        plt.tight_layout()
        
        # Compile results
        results = {
            'num_intersections': len(intersections),
            'intersection_density': density,
            'mean_angle': np.mean(angles) if angles else 0,
            'std_angle': np.std(angles) if angles else 0
        }
        
        return results

# Example usage
if __name__ == "__main__":
    # Generate example path data with noise
    t = np.linspace(0, 4*np.pi, 200)
    # Create a figure-8 pattern with significant noise
    x = 10 * np.sin(t) + np.random.normal(0, 1.0, len(t))
    y = 10 * np.sin(2*t) + np.random.normal(0, 1.0, len(t))
    
    # Create analyzer and smooth the path
    analyzer = PathAnalyzer(smooth_window=15)
    x_smooth, y_smooth = analyzer.smooth_path(x, y, method='savgol')
    
    # Analyze both original and smoothed paths
    plt.figure(figsize=(15, 5))
    
    # Original path
    plt.subplot(131)
    plt.plot(x, y, 'b-', alpha=0.6, label='Original')
    plt.title('Original Path')
    plt.axis('equal')
    plt.legend()
    
    # Smoothed path
    plt.subplot(132)
    plt.plot(x_smooth, y_smooth, 'g-', alpha=0.6, label='Smoothed')
    plt.title('Smoothed Path')
    plt.axis('equal')
    plt.legend()
    
    # Analyze smoothed path
    results = analyzer.analyze_and_plot(x_smooth, y_smooth, "Intersection Analysis (Smoothed Path)")
    plt.gcf().set_size_inches(15, 5)
    
    print("\nAnalysis Results:")
    print(f"Number of intersections: {results['num_intersections']}")
    print(f"Intersection density: {results['intersection_density']:.3f}")
    print(f"Mean intersection angle: {results['mean_angle']:.1f}°")
    print(f"Std of intersection angles: {results['std_angle']:.1f}°")
    
    plt.show()