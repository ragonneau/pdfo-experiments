from profiles import Profiles

if __name__ == '__main__':
    profiles = Profiles(1, 10, "unconstrained")
    profiles(["newuoa", "bobyqa", "lincoa", "cobyla", "uobyqa"])
    del profiles

    profiles = Profiles(1, 50, "unconstrained")
    profiles(["newuoa", "bobyqa", "lincoa", "cobyla"])
    del profiles

    profiles = Profiles(1, 50, "unconstrained", feature="noisy", noise_level=1e-2)
    profiles(["newuoa", "bobyqa", "lincoa", "cobyla"])
    del profiles
