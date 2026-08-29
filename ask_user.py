def ask_noise():
    while True:
        try:
            # Asks the user a probability between 0 and 1
            p = float(input("Flip percentage (0-100): "))

            if 0 <= p <= 100:
                return p

        except ValueError:
            pass

        # Shows the same message both for non-numbers and numbers not in [0,100]
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
    try:
        beta = float(input("Parameter beta (>1): "))
        if beta < 1:
            print("Choose a number greater than 1")
        else:
            return beta
    except ValueError:
        print("Choose a number greater than 1")