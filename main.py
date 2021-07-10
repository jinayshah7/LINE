from utils import svm_classify, batchgen_train, LINE_loss, load_data
import numpy as np
from model import create_model
import random


if __name__ == "__main__":
    vector_tag = ""

    label_file = vector_tag + '.line_labels'
    edge_file = vector_tag + '.line_edges'
    save_file_name = vector_tag + '.vector'
    epoch_num = 1
    factors = 512
    batch_size = 1000
    negative_sampling = "UNIFORM" # UNIFORM or NON-UNIFORM
    negativeRatio = 5
    split_ratios = [0.6, 0.2, 0.2]
    svm_C = 0.1

    np.random.seed(2017)
    random.seed(2017)

    adj_list, labels_dict = load_data(label_file, edge_file)
    epoch_train_size = (((int(len(adj_list)/batch_size))*(1 + negativeRatio)*batch_size) + (1 + negativeRatio)*(len(adj_list)%batch_size))
    numNodes = np.max(adj_list.ravel()) + 1
    data_gen = batchgen_train(adj_list, numNodes, batch_size, negativeRatio, negative_sampling)

    model, embed_generator = create_model(numNodes, factors)
    model.summary()

    model.compile(optimizer='rmsprop', loss={'left_right_dot': LINE_loss})

    model.fit_generator(data_gen, steps_per_epoch=epoch_train_size, epochs=epoch_num, verbose=1)

    new_X = []
    new_label = []
    vectors = []
    keys = list(labels_dict.keys())
    np.random.shuffle(keys)

    for k in keys:
        v = labels_dict[k]
        x = embed_generator.predict_on_batch([np.asarray([k]), np.asarray([k])])
        vector = x[0][0] + x[1][0]
        vector_string = ""
        for i in vector:
          vector_string += str(i) + " "
        
        vectors.append(f"{v} {vector_string}")
        new_label.append(labels_dict[k])
    for i in vectors:
      print(i)

    with open(save_file_name, "w") as f:
      for i in vectors:
        f.write(i+"\n")
