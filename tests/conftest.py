import hypothesis
import torch as pt

hypothesis.settings.register_profile("default", deadline=None)
hypothesis.settings.register_profile("ci", max_examples=100, deadline=None)
hypothesis.settings.register_profile("more", max_examples=1000, deadline=None)
hypothesis.settings.load_profile("default")

pt.set_grad_enabled(False)
if pt.cuda.is_available():
    pt.set_default_device("cuda:0")
