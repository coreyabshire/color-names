

ps = np.array([[ 0.0,  0.0,  1.0],
               [ 0.0,  1.0, -1.0],
               [ 0.0, -1.0, -1.0],
               [-1.0,  0.0,  1.0]])
d = Delaunay(ps)

ps = np.array([[ 0.0, -1.0,  1.0],
               [ 0.0,  1.0,  1.0],
               [ 0.0,  1.0, -1.0],
               [ 0.0, -1.0, -1.0],
               [-1.0,  0.0,  0.0]])
d = Delaunay(ps)
