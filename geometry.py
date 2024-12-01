import numpy as np
import depthai as dai
import cv2

class Point:
    def __init__(self, x: float, y: float) -> None:
        """Create a 2D point with coordinates (x, y)

        Parameters
        ----------
        x : float
        y : float
        """
        __slots__ = ["x", "y"]
        self.x: float = x
        self.y: float = y
        
    @property
    def intx(self) -> int:
        """int(x)"""
        return int(self.x)
    
    @property
    def inty(self) -> int:
        """int(y)"""
        return int(self.y)

    def list2f(self) -> list[float, float]:
        """float coordinates list"""
        return [self.x, self.y]
    
    def listint(self) -> list[int, int]:
        """int coordinates list"""
        return [int(self.x), int(self.y)]
    
    def tuple2f(self) -> tuple[float, float]:
        """float coordinates tuple"""
        return (self.x, self.y)
    
    def tupleint(self) -> tuple[int, int]:
        """int coordinates tuple"""
        return (int(self.x), int(self.y))
    
    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.x:.2f}, {self.y:.2f})'
    

class RotatedRect:
    def __init__(self, bbox: tuple[Point, Point], angle: float) -> None:
        """Create a rotated rectangle

        Parameters
        ----------
        bbox : tuple[Point, Point]
            It's non rotated bottom left and top right corners
        angle : float
            Angle of rotation in couterclockwise direction
        """
        # 
        #   *-----------------------* 
        #   |           |           |
        #   |           |           | halfheight
        #   |           | centre    |
        #   |-----------X-----------|--
        #   |           |           |
        #   |           |           | halfheight
        #   |           |           |
        #   *-----------------------*
        #     halfwidth | halfwidth
        
        pt1, pt2 = bbox
        self.centre: Point = Point((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2)
        self.halfwidth: float = abs(pt1.x - pt2.x) / 2
        self.halfheight: float = abs(pt1.y - pt2.y) / 2
        self.angle: float = -1 * angle
        
    
    def get_rotated_points(self) -> np.array:
        """Returns coordinates of corners of a rotated rectangle

        Returns:
            list (Point, Point, Point, Point): list of 4 corners
        """
        #  D                       C
        #   *---------------------* 
        #   |                     |
        #   |                     |
        #   |                     |
        #   |                     |
        #   |                     |
        #   |                     |
        #   *---------------------*
        # A                         B
        #
        

        cos = np.cos(self.angle)
        sin = np.sin(self.angle)
        
        # operations below perform rectangle rotation around its centre
        matrix = np.array([[cos,    -sin,   self.centre.x*(1-cos) + self.centre.y*sin ],
                           [sin,     cos,   self.centre.y*(1-cos) - self.centre.x*sin ],
                           [ 0,       0,                          1                   ]])

        A: Point = Point(self.centre.x - self.halfwidth, self.centre.y + self.halfheight)
        tmp_A: np.array = np.array([[A.intx], [A.inty], [1]])
        tmp_A_prime: np.ndarray = np.dot(matrix, tmp_A)
        A_prime: Point = Point(tmp_A_prime[0][0], tmp_A_prime[1][0])
        del A, tmp_A, tmp_A_prime
        
        B: Point = Point(self.centre.x + self.halfwidth, self.centre.y + self.halfheight)
        tmp_B: np.array = np.array([[B.intx], [B.inty], [1]])
        tmp_B_prime: np.ndarray = np.dot(matrix, tmp_B)
        B_prime: Point = Point(tmp_B_prime[0][0], tmp_B_prime[1][0])
        del B, tmp_B, tmp_B_prime
        
        C: Point = Point(self.centre.x + self.halfwidth, self.centre.y - self.halfheight)
        tmp_C: np.array = np.array([[C.intx], [C.inty], [1]])
        tmp_C_prime: np.ndarray = np.dot(matrix, tmp_C)
        C_prime: Point = Point(tmp_C_prime[0][0], tmp_C_prime[1][0])
        del C, tmp_C, tmp_C_prime
        
        D: Point = Point(self.centre.x - self.halfwidth, self.centre.y - self.halfheight)
        tmp_D: np.array = np.array([[D.intx], [D.inty], [1]])
        tmp_D_prime: np.ndarray = np.dot(matrix, tmp_D)
        D_prime: Point = Point(tmp_D_prime[0][0], tmp_D_prime[1][0])
        del D, tmp_D, tmp_D_prime
    
        return np.array([A_prime.listint(), B_prime.listint(), C_prime.listint(), D_prime.listint()], dtype=np.int32)

    
    def get_depthai_RotatedRect(self) -> dai.RotatedRect:
        """Retrieve depthai.RotatedRect"""
        # dai.RotatedRect is in format 
        # [[x_centre, y_centre], [width, height], angle]
        rr: dai.RotatedRect = dai.RotatedRect()
        rr.center.x   = self.centre.x
        rr.center.y   = self.centre.y
        rr.size.width = 2 * self.halfwidth
        rr.size.hight = 2 * self.halfheight
        rr.angle = self.angle
        return rr


    def get_cv_RotatedRect(self) -> cv2.RotatedRect:
        """Retrieve cv2.RotatedRect"""
        return cv2.RotatedRect(self.centre.listint(), [2 * self.halfwidth, 2 * self.halfheight],  np.rad2deg(self.angle))
    
    def __str__(self) -> str:
        return f'RotatedRect(centre={self.centre}, width={2*self.halfwidth}, height={2*self.halfheight} angle={self.angle})'