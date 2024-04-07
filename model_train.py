import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
from utils import action_map_small, gen_sequence, get_all_possible_actions_cube_small, chunker, \
    flatten_1d_b
from model import get_model

if __name__ == "__main__":

    N_SAMPLES = 100
    N_EPOCH = 100

    checkpoint = ModelCheckpoint("auto.keras", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    early = EarlyStopping(monitor="val_loss", mode="min", patience=1000)

    reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.1, patience=50, min_lr=1e-8)

    callbacks_list = [checkpoint, early, reduce_on_plateau]

    model = get_model(lr=0.0001)
    # model.load_weights("auto.weights.h5")

    for i in range(N_EPOCH):
        print("epochs: " +  str(i) + "/" + str(N_EPOCH))
        cubes = []
        distance_to_solved = []
        for j in tqdm(range(N_SAMPLES)):
            _cubes, _distance_to_solved = gen_sequence(25)
            cubes.extend(_cubes)
            distance_to_solved.extend(_distance_to_solved)

        cube_next_reward = []
        flat_next_states = []
        cube_flat = []

        for c in tqdm(cubes):
            flat_cubes, rewards = get_all_possible_actions_cube_small(c)
            cube_next_reward.append(rewards)
            flat_next_states.extend(flat_cubes)
            cube_flat.append(flatten_1d_b(c))

        for _ in range(20):

            cube_target_value = []
            cube_target_policy = []

            next_state_value, _ = model.predict(np.array(flat_next_states), batch_size=1024)
            next_state_value = next_state_value.ravel().tolist()
            next_state_value = list(chunker(next_state_value, size=len(action_map_small)))

            for c, rewards, values in tqdm(zip(cubes, cube_next_reward, next_state_value)):
                r_plus_v = 0.4*np.array(rewards) + np.array(values)
                target_v = np.max(r_plus_v)
                target_p = np.argmax(r_plus_v)
                cube_target_value.append(target_v)
                cube_target_policy.append(target_p)

            cube_target_value = (cube_target_value-np.mean(cube_target_value))/(np.std(cube_target_value)+0.01)

            print(cube_target_policy[-30:])
            print(cube_target_value[-30:])

            sample_weights = 1. / np.array(distance_to_solved)
            sample_weights = sample_weights * sample_weights.size / np.sum(sample_weights)
            model.fit(np.array(cube_flat), [np.array(cube_target_value), np.array(cube_target_policy)[..., np.newaxis]],
                      epochs=1, batch_size=128, sample_weight=[sample_weights, sample_weights])
            sample_weight=[sample_weights, sample_weights],

        model.save_weights("auto.weights.h5")
