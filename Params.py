def get_run_parameters():
    batch_size = 90
    pic_width = 128
    prc_patterns = 10
    n_gray_levels = 255
    m_patterns = (pic_width ** 2) * prc_patterns // 100

    initial_lr = 10 ** -3
    div_factor_lr = 10
    num_dif_lr = 3
    n_epochs = 20
    lr_vector = [10**-3, 10**-4, 10-6]
    epochs_vector = [50, 10, 140]


    input_shape = (pic_width, pic_width, batch_size)
    num_train_samples, num_test_samples = 504, 50

    return batch_size, pic_width, prc_patterns, n_gray_levels, m_patterns, initial_lr, div_factor_lr, num_dif_lr,\
           n_epochs, num_train_samples, num_test_samples, input_shape, lr_vector, epochs_vector