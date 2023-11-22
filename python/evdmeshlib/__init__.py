from evdmeshlib._rs.linalg import Vec3


def test_vec3():
    v = Vec3(1, 2, 3)
    w = Vec3(4, 5, 6)

    print(v)
    print(w)
    print(v + w)


if __name__ == '__main__':
    test_vec3()
