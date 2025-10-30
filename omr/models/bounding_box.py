from pydantic import BaseModel


class BoundingBox(BaseModel):
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def x_left(self):
        return self.x_center - (self.width / 2)

    @property
    def x_right(self):
        return self.x_center + (self.width / 2)

    @property
    def y_bottom(self):
        return self.y_center - (self.height / 2)

    @property
    def y_top(self):
        return self.y_center + (self.height / 2)
