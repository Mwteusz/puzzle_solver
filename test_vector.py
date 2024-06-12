from teeth_detection import Vector

if __name__ == '__main__':
    offset=-20
    offset2=20
    vector: Vector = Vector((0+offset, 0+offset2), (10+offset, 4+offset2))
    print(1,vector.normalise_extreme())

    vector: Vector = Vector((0+offset, 0+offset2), (-10+offset, -4+offset2))
    print(2,vector.normalise_extreme())

    vector: Vector = Vector((0+offset, 0+offset2), (-10+offset, 4+offset2))
    print(3,vector.normalise_extreme())

    vector: Vector = Vector((0+offset, 0+offset2), (10+offset, -4+offset2))
    print(4,vector.normalise_extreme())

    vector: Vector = Vector((0+offset, 0+offset2), (10+offset, -10+offset2))
    print(5,vector.normalise_extreme())

    vector: Vector = Vector((0+offset, 0+offset2), (-1+offset,-5+offset2))
    print(6,vector.normalise_extreme())

    vector: Vector = Vector((0+offset, 0+offset2), (-2+offset, 3+offset2))
    print(7,vector.normalise_extreme())