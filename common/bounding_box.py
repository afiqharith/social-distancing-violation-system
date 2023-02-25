class BoundingBox:
    """
        (x, y)
                -----------
                |         |
                |         |
                |         | height
                |         |
                |         |
                -----------
                   width
    """
    NON_VIOLATE = 0x00
    VIOLATE = 0x01

    def __init__(self, x, y, width, height, is_violate = None) -> None:
        self.xmin = x
        self.ymin = y
        self.xmax = (x + width)
        self.ymax = (y + height)
        self.is_violate = is_violate

    def get_groundplane_center_point(self) -> tuple:
        return (
            int((self.xmax + self.xmin)/2), 
            int(self.ymax))

    def get_center_axis_x(self) -> int:
        return (self.xmin + self.xmax)/2

    def get_center_axis_y(self) -> int:
        return (self.ymin + self.ymax)/2
