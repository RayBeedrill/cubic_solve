import numpy as np
from utils import parse_cube, action_map, flatten_1d_b, perc_solved_cube

def nn_solve_cube(encoding, model):
    sample_X, sample_Y, cubes = parse_cube(encoding)
    cube = cubes[0]
    cube.score = 0
    print(cube)
    list_sequences = [[cube]]

    existing_cubes = set()

    for j in range(200):
        print("step: " + str(j))
        X = [flatten_1d_b(x[-1]) for x in list_sequences]

        value, policy = model.predict(np.array(X), batch_size=1024)

        new_list_sequences = []

        for x, policy in zip(list_sequences, policy):
            pred = np.argsort(policy)

            cube_1 = x[-1].copy()(list(action_map.keys())[pred[-1]])
            cube_2 = x[-1].copy()(list(action_map.keys())[pred[-2]])

            new_list_sequences.append(x + [cube_1])
            new_list_sequences.append(x + [cube_2])

        print("new_list_sequences", len(new_list_sequences))
        last_states_flat = [flatten_1d_b(x[-1]) for x in new_list_sequences]
        value, _ = model.predict(np.array(last_states_flat), batch_size=1024)
        value = value.ravel().tolist()
        for x, v in zip(new_list_sequences, value):
            x[-1].score = v if str(x[-1]) not in existing_cubes else -1

        new_list_sequences.sort(key=lambda x: x[-1].score , reverse=True)

        new_list_sequences = new_list_sequences[:100]

        existing_cubes.update(set([str(x[-1]) for x in new_list_sequences]))

        list_sequences = new_list_sequences

        list_sequences.sort(key=lambda x: perc_solved_cube(x[-1]), reverse=True)

        prec = perc_solved_cube((list_sequences[0][-1]))

        print(prec)

        if prec == 1:
            break

    print(perc_solved_cube(list_sequences[0][-1]))
    print(list_sequences[0][-1])
