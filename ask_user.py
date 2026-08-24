def ask_noise():
    while True:
        try:
            # Asks the user a probability between 0 and 1
            p = float(input("Flip probability (0-1): "))

            if 0 <= p <= 1:
                return p

        except ValueError:
            pass

        # Shows the same message both for non-numbers and numbers not in [0,1]
        print("Enter a number between 0 and 1")


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