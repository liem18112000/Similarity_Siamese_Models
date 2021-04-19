import numpy as np
import matplotlib.pyplot as plt

def preprocess_data_into_groups(train, test, image_size, channel=1):
    x_train, y_train = train
    x_test, y_test = test
    x_train = x_train.reshape(-1, image_size, image_size, channel).astype('float32') / 255.
    x_test = x_test.reshape(-1, image_size, image_size, channel).astype('float32') / 255.
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    print('Training', x_train.shape, x_train.max())
    print('Testing', x_test.shape, x_test.max())
    y_train = y_train.reshape((-1, ))
    y_test = y_test.reshape((-1, ))
    print('Testing', y_train.shape)
    print('Testing', y_test.shape)

    # reorganize by groups
    train_groups = [x_train[np.where(y_train == i)[0]]
                    for i in np.unique(y_train)]
    test_groups = [x_test[np.where(y_test == i)[0]]
                   for i in np.unique(y_train)]
    print('train groups:', [x.shape[0] for x in train_groups])
    print('test groups:', [x.shape[0] for x in test_groups])

    return train_groups, test_groups


def estimate_conv_layers(num):
    estimator = 1
    while num >= 2 ** estimator:
        estimator += 1
    return estimator - 1


def gen_random_batch(in_groups, batch_halfsize=8):
    out_img_a, out_img_b, out_score = [], [], []
    all_groups = list(range(len(in_groups)))
    for match_group in [True, False]:
        group_idx = np.random.choice(all_groups, size=batch_halfsize)
        out_img_a += [in_groups[c_idx]
                      [np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in group_idx]
        if match_group:
            b_group_idx = group_idx
            out_score += [1]*batch_halfsize
        else:
            # anything but the same group
            non_group_idx = [np.random.choice(
                [i for i in all_groups if i != c_idx]) for c_idx in group_idx]
            b_group_idx = non_group_idx
            out_score += [0]*batch_halfsize

        out_img_b += [in_groups[c_idx]
                      [np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in b_group_idx]

    return np.stack(out_img_a, 0), np.stack(out_img_b, 0), np.stack(out_score, 0)


def show_model_output(model, test_groups, nb_examples=3):
    pv_a, pv_b, pv_sim = gen_random_batch(test_groups, nb_examples)
    pred_sim = model.predict([pv_a, pv_b])
    fig, m_axs = plt.subplots(2, pv_a.shape[0], figsize=(
        5 * nb_examples, 1.75 * nb_examples))
    for c_a, c_b, c_d, p_d, (ax1, ax2) in zip(pv_a, pv_b, pv_sim, pred_sim, m_axs.T):
        ax1.imshow(c_a[:, :, 0])
        ax1.set_title('Image A\n Actual: %3.0f%%' % (100*c_d))
        ax1.axis('off')
        ax2.imshow(c_b[:, :, 0])
        ax2.set_title('Image B\n Predicted: %3.0f%%' % (100*p_d))
        ax2.axis('off')
    return fig


def show_images_in_groups(train_groups, nb_examples=3):
    pv_a, pv_b, pv_sim = gen_random_batch(train_groups, nb_examples)
    fig, m_axs = plt.subplots(2, pv_a.shape[0], figsize=(
        5 * nb_examples, 1.75 * nb_examples))
    for c_a, c_b, c_d, (ax1, ax2) in zip(pv_a, pv_b, pv_sim, m_axs.T):
        ax1.imshow(c_a[:, :, 0])
        ax1.set_title('Image A')
        ax1.axis('off')
        ax2.imshow(c_b[:, :, 0])
        ax2.set_title('Image B\n Similarity: %3.0f%%' % (100*c_d))
        ax2.axis('off')

# make a generator out of the data
def siam_gen(in_groups, batch_size=1024):
    while True:
        pv_a, pv_b, pv_sim = gen_random_batch(in_groups, batch_size//2)
        yield [pv_a, pv_b], pv_sim


def plot_triplets(examples, image_size):
    plt.figure(figsize=(6, 2))
    for i in range(3):
        plt.subplot(1, 3, 1 + i)
        plt.imshow(np.reshape(
            examples[i], (image_size, image_size)), cmap='binary')
        plt.xticks([])
        plt.yticks([])
    plt.show()


def create_batch(x_train, y_train, image_size, batch_size=1024):
    x_anchors = np.zeros((batch_size, image_size, image_size, 1))
    x_positives = np.zeros((batch_size, image_size, image_size, 1))
    x_negatives = np.zeros((batch_size, image_size, image_size, 1))

    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = np.random.randint(0, x_train.shape[0] - 1)
        x_anchor = x_train[random_index]
        y = y_train[random_index]

        indices_for_pos = np.squeeze(np.where(y_train == y))
        indices_for_neg = np.squeeze(np.where(y_train != y))

        x_positive = x_train[indices_for_pos[np.random.randint(
            0, len(indices_for_pos) - 1)]]
        x_negative = x_train[indices_for_neg[np.random.randint(
            0, len(indices_for_neg) - 1)]]

        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative

    return [x_anchors, x_positives, x_negatives]


def data_generator(x_train, y_train, image_size, emb_size=512, batch_size=1024):
    while True:
        x = create_batch(x_train, y_train, image_size, batch_size)
        y = np.zeros((batch_size, 3*emb_size))
        yield x, y
