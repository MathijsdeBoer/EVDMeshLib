from evdmeshlib.linalg import Vec3


def main():
    vec = Vec3(1, 2, 3)
    other = Vec3(4, 5, 6)

    print(f"vec: {vec}")
    print(f"other: {other}")
    print(f"vec + other: {vec + other}")


if __name__ == "__main__":
    main()
