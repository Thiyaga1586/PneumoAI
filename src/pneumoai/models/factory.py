class DummyModel:
    def predict_proba(self, image_array):
        return 0.82


def build_model(version: str):
    return DummyModel()