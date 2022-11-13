class FunctionalModelWrapper:
    """
    Wraps a functional model constructor in a class to be recognized in BaseModel
    """

    def __init__(self, class_function):
        self.class_function = class_function

    def __call__(self, *args, **kwargs):
        return self.class_function(*args, **kwargs)
