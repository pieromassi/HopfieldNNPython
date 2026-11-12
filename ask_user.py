def ask_noise():
    while True:
        try:
            p = float(input("Flip percentage (0-100): "))
            if 0 <= p <= 100:
                return p
        except ValueError:
            pass
        print("Enter a number between 0 and 100")


def ask_crop(size):
    while True:
        try:
            height = int(input(f"Height (1-{size}): "))
            width = int(input(f"Width (1-{size}): "))
            if not (1 <= height <= size and 1 <= width <= size):
                print("Out of range, try again")
                continue
        except ValueError:
            print("Not a valid integer, try again")
            continue      
        return height, width


def ask_mode():
    while True:
        mode = input("Corruption mode (noise/crop): ").strip().lower()
        if mode in ("noise", "crop"):
            return mode
        else: 
            print("Choose between noise and crop")         


def ask_field():
    while True:
        field = input("Field mode (hebb/cubic/exp): ").strip().lower()
        if field in ("hebb", "cubic", "exp"):
            return field
        else:
            print("Choose between hebb, cubic and exp")


def ask_self_int():
    while True:
        self_int = input("Self interaction (yes/no): ").strip().lower()
        if self_int == "yes":
            return True
        elif self_int == "no":
                return False
        else:
            print("Choose between yes and no")


def ask_beta():
    while True:
        try:
            beta = float(input("Parameter beta (>1): "))
            if beta < 1:
                print("Choose a number greater than 1")
                continue
            else:
                return beta
        except ValueError:
            print("Choose a number greater than 1")
            continue


def ask_n_per_person():
    while True:
        try:
            n = int(input("Faces per person (1/10): "))
            if 1 <= n <= 10:
                return n
            else:
                print("Choose a number between 1 and 10")
                continue
        except ValueError:
            print("Choose a number between 1 and 10")
            continue


def ask_dynamics():
    while True:
        dynamics = input(
            "Dynamics (deterministic/metropolis-hastings/simulated annealing): "
        ).strip().lower()
        dynamics = dynamics.replace("-", "_").replace(" ", "_")

        if dynamics in (
            "deterministic",
            "metropolis_hastings",
            "simulated_annealing",
        ):
            return dynamics

        print(
            "Choose between deterministic, metropolis-hastings "
            "and simulated annealing"
        )

def ask_temperature():
    while True:
        try:
            T = float(input("Temperature (>0): "))
            if T <= 0:
                print("Choose a number greater than 0")
                continue
            else:
                return T
        except ValueError:
            print("Choose a number greater than 0")
            continue