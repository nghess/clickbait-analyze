import cv2

class GridMaze: 
    def __init__(self, maze_bounds, maze_dims):
            assert len(maze_bounds) == 2, "Maze bounds must be 2-dimensional."
            assert len(maze_dims) == 2, "Maze dimensions must be 2-dimensional (rows * columns)."

            self.bounds = maze_bounds
            self.shape = maze_dims
            
            cellsize_x = maze_bounds[0] // maze_dims[0]
            cellsize_y = maze_bounds[1] // maze_dims[1]
            
            # Generate Grid
            self.cells = [
                ((x * cellsize_x, y * cellsize_y), 
                ((x + 1) * cellsize_x, (y + 1) * cellsize_y))
                for y in range(self.shape[1])
                for x in range(self.shape[0])
            ]

def draw_grid_cell(frame, pt1, pt2, color, thickness, opacity):
    # Create a separate image for the rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, color, thickness)
    
    # Blend the overlay with the original frame
    return cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)